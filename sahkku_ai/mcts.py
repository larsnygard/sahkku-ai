"""MCTS for sáhkku: batched leaf-parallel PUCT over the CURRENT turn.

Design decisions (they are the whole game):

* **Dice are chance, but only between turns.** The search runs over the moves
  of the turn that is being played (the dice are already thrown and known, and
  the turn is at most three moves). A node whose move ends the turn becomes a
  LEAF and is evaluated by the value head — the unknown next roll is therefore
  implicit in the value, and the tree never needs chance nodes. (Optional
  root-roll sampling is left as a knob for later.)
* **Value, not playouts.** Leaves are scored by the value head; the search does
  not play the game to the end. That is what makes it affordable: the current
  8-rollout scheme burns ~52 plies per rollout, a 400-simulation search burns
  one forward per leaf.
* **Leaf-parallel batching + virtual loss.** One forward per batch of leaves,
  otherwise the 2 ms kernel-launch-bound forward is paid serially 400 times.
* **Signs flip per side-to-move change, not per ply** — inside a turn the same
  side keeps acting, so a naive per-ply flip would invert values.

Targets produced: π = visit counts at the root (temperature-controlled),
z = root value from the search (or the game outcome, caller's choice).
"""
from __future__ import annotations
import math
import numpy as np

from . import engine as E
from . import features as FT
from .selfplay import softmax_prior, dirichlet


class Node:
    __slots__ = ("st", "P", "N", "W", "kids", "parent", "aid", "value",
                 "pending", "done", "mvmap", "vN", "vW")

    def __init__(self, st, parent=None, aid=None):
        self.st = st
        self.P = {}
        self.N = {}
        self.W = {}
        self.kids = {}
        self.parent = parent
        self.aid = aid
        self.value = 0.0
        self.pending = False
        self.done = False
        self.mvmap = {}          # aid -> move (cache: legal_ids er dyr)
        # Virtuelt tap: medan ei bunt løv vert evaluert på GPU-en, må
        # seleksjonen tru at vegane alt er vitja — elles vel han same sti
        # om att og att, og bunten vert aldri større enn éin.
        self.vN = {}
        self.vW = {}

    def visits(self):
        return sum(self.N.values())


def _move_of(st, aid):
    for a, mv in FT.legal_ids(st):
        if a == aid:
            return mv
    return None


def _apply(st, aid, rng):
    """Legg trekket på stillinga (settle med, so turen kan ta slutt)."""
    for a, mv in FT.legal_ids(st):
        if a != aid:
            continue
        if mv.get("kind") == "pass":
            E.pass_remaining(st)
        elif mv.get("kind") == "reroll":
            E.reroll_x(st, rng.random)
            st.phase = "move"
        else:
            E.apply_move(st, mv)
            E.settle(st)
        return True
    return False


def _priors(st, pol, noise_alpha=0.3, noise_eps=0.25, rng=None, mvmap=None):
    ids = FT.legal_ids(st)
    if mvmap is not None:
        mvmap.update(ids)
    legal = [a for a, _ in ids]
    if not legal:
        return {}, 0.0
    # Å SØKJE på omkast er meiningslaust: det er ei null-handling som berre
    # kastar terningane på nytt (tilfeldig, utanfor treet). Vart ho med, fekk
    # ho ~86 % av vitjingane og partia sto på 300-ply-taket. Ho er framleis
    # lovleg — men berre når ho er det einaste ein kan gjere.
    real = [a for a in legal if a != FT.SPEC_BASE + 1]
    if real:
        legal = real
    prior = softmax_prior(pol, legal)
    P = {a: p for a, p in zip(legal, prior)}
    if noise_alpha > 0 and rng is not None and len(legal) > 1:
        d = dirichlet(rng, noise_alpha, len(legal))
        for a, dv in zip(legal, d):
            P[a] = (1 - noise_eps) * P[a] + noise_eps * dv
    return P, max(prior)


def _terminal_value(st):
    """±1 om partiet er avgjort (frå sida-som-skal-trekke sin POV), elles None."""
    if st.winner is None:
        return None
    return 1.0 if st.winner == st.turn else -1.0


VLOSS = 1.0        # virtuelt tap per steg i ein bunt


def _select(root, c_puct, blocked=None):
    """Gå nedover med PUCT til eit løv som kan utvidast.

    Returnerer (løv, sti). `sti` er [(node, aid), …] frå rota, og kvart steg
    har fått eit virtuelt tap pålagt: N er telja opp og W ned med VLOSS, so
    neste seleksjon i same bunt ikkje går same vegen. Tapa vert fjerna i
    `_unloss` når den verkelege verdien er komen attende frå nettet.
    """
    node = root
    path = []
    while True:
        if node.done or node.pending or not node.P:
            return node, path
        sq = math.sqrt(node.visits() + 1.0)
        best_a, best_u = None, -1e18
        for a, p in node.P.items():
            child = node.kids.get(a)
            if child is not None and child.pending:
                continue
            if blocked and (id(node), a) in blocked:
                continue          # denne vegen stod fast i denne runda
            n = node.N.get(a, 0) + node.vN.get(a, 0)
            w = node.W.get(a, 0.0) - node.vW.get(a, 0.0)
            q = (w / n) if n else 0.0
            u = q + c_puct * p * sq / (1.0 + n)
            if u > best_u:
                best_u, best_a = u, a
        if best_a is None:            # alle born er opptekne/ferdige
            return node, path
        child = node.kids.get(best_a)
        if child is None:
            mv = node.mvmap.get(best_a)
            if mv is None:
                node.done = True
                node.value = _terminal_value(node.st) or 0.0
                return node, path
            child_st = node.st.clone()
            _apply_mv(child_st, mv, _RNG_HOLDER["rng"])
            child = Node(child_st, parent=node, aid=best_a)
            node.kids[best_a] = child
            node.N.setdefault(best_a, 0)
            node.W.setdefault(best_a, 0.0)
            # legg virtuelt tap på steget ned til det nye løvet òg
            node.vN[best_a] = node.vN.get(best_a, 0) + 1
            node.vW[best_a] = node.vW.get(best_a, 0.0) + VLOSS
            path.append((node, best_a))
            return child, path
        node.vN[best_a] = node.vN.get(best_a, 0) + 1
        node.vW[best_a] = node.vW.get(best_a, 0.0) + VLOSS
        path.append((node, best_a))
        node = child


def _unloss(path):
    """Fjern dei virtuelle tapa før den verkelege verdien vert ført oppover."""
    for nd, a in path:
        if nd.vN.get(a):
            nd.vN[a] -= 1
            if not nd.vN[a]:
                nd.vN.pop(a, None)
        if nd.vW.get(a):
            nd.vW[a] = max(0.0, nd.vW[a] - VLOSS)
            if nd.vW[a] == 0.0:
                nd.vW.pop(a, None)


def _apply_mv(st, mv, rng):
    """Legg eit alt funne trekk på stillinga (inga oppramsing på nytt)."""
    if mv.get("kind") == "pass":
        E.pass_remaining(st)
    elif mv.get("kind") == "reroll":
        E.reroll_x(st, rng.random)
        st.phase = "move"
    else:
        E.apply_move(st, mv)
        E.settle(st)


_RNG_HOLDER = {"rng": None}


def _backup(node):
    """Rull verdiar oppover; snu forteikn når sida-som-trekkjer skiftar."""
    v = node.value
    cur = node
    while cur.parent is not None:
        par = cur.parent
        pv = v if par.st.turn == cur.st.turn else -v
        par.N[cur.aid] = par.N.get(cur.aid, 0) + 1
        par.W[cur.aid] = par.W.get(cur.aid, 0.0) + pv
        v = pv
        cur = par


def search(pred, st, cfg, rng, root_noise=True):
    """Køyr MCTS frå `st`. Returnerer (π: aid->sannsyn, z: rot-verdi)."""
    sims = int(cfg["sims"])
    c_puct = float(cfg.get("c_puct", 1.6))
    batch = max(1, int(cfg.get("leaf_batch", 8)))
    temp = float(cfg.get("temp", 1.0))
    _RNG_HOLDER["rng"] = rng

    root = Node(st)
    pol, val = pred.predict([st])
    root.P, _ = _priors(st, pol[0],
                        cfg.get("noise_alpha", 0.3) if root_noise else 0.0,
                        cfg.get("noise_eps", 0.25), rng, root.mvmap)
    if not root.P:                      # inga lovleg handling: ingenting å søkje
        return {}, 0.0
    root.value = float(val[0])

    done_sims = 0
    while done_sims < sims:
        todo = min(batch, sims - done_sims)
        pending, immediate = [], []
        blocked = set()          # (foreldre-id, handling) som ikkje fører til nytt løv
        _dbg = cfg.get("_dbg")
        if _dbg:
            print(f"  [runde] todo={todo} done_sims={done_sims}", flush=True)
        for _ in range(todo):
            leaf, path = _select(root, c_puct, blocked)
            if leaf.pending:                 # alt i ein bunt som ventar
                if _dbg: print("     -> alt ventar", flush=True)
                _unloss(path)
                continue
            if leaf.done:
                # Eit alt avgjort løv (t.d. vinnande kapsel) vert valt på nytt
                # kvar sim — og må krediterast kvar gong, elles står N på 1
                # medan resten av budsjettet renn ut i inkje, og π (som er
                # vitjingsbasert) viser feil favoritt.
                _unloss(path)
                _backup(leaf)
                immediate.append(leaf)
                continue
            if leaf.P and leaf.kids and all((ch.pending or ch.done)
                                            for ch in leaf.kids.values()):
                if _dbg: print("     -> born ventar", flush=True)
                # Denne vegen står fast i denne runda (berre ventande born).
                # Sperr steget ned til henne, so neste seleksjon prøver ein
                # SØSKEN i staden — det er difor bunten før vart storleik 1.
                if path:
                    blocked.add((id(path[-1][0]), path[-1][1]))
                _unloss(path)
                continue
            tv = _terminal_value(leaf.st)
            if tv is not None:
                leaf.value, leaf.done = tv, True
                _unloss(path)
                # AVGJORT LØV MÅ FØRAST OPPOVER. Utan denne linja vart ein
                # vinnande konge-/dronningkapsel aldri kreditert: verdien
                # vart sett, men N/W hjå forelderen stod på 0, so søket
                # føretrekte eit roleg trekk framfor å ta dronninga og vinne.
                _backup(leaf)
                immediate.append(leaf)
                continue
            if _dbg: print("     -> NYTT LØV", flush=True)
            leaf.pending = True
            pending.append((leaf, path))
            done_sims += 1
        if pending:
            pols, vals = pred.predict([n.st for n, _ in pending])
            for (n, path_), p, v in zip(pending, pols, vals):
                n.P, _ = _priors(n.st, p, 0.0, 0.0, None, n.mvmap)
                n.value = float(v)
                n.pending = False
                if not n.P:                      # inga lovleg handling → dødt løv
                    n.done = True
                    n.value = 0.0
                _unloss(path_)                   # fjern tapa FØR verdien vert ført opp
                _backup(n)
        done_sims += len(immediate)
        if not pending and not immediate:
            break

    if cfg.get("_debug"):
        print("[mcts] rot:", {a: {"N": root.N.get(a, 0), "W": round(root.W.get(a, 0.0), 2),
                                "done": (root.kids.get(a).done if root.kids.get(a) else None),
                                "har_born": (len(root.kids.get(a).kids) if root.kids.get(a) else None)}
                             for a in root.P}, flush=True)
    total = root.visits()
    if total == 0:
        return {a: p for a, p in root.P.items()}, root.value
    pi = {}
    for a, n in root.N.items():
        if n <= 0:
            continue
        pi[a] = n ** (1.0 / max(temp, 1e-3)) if temp > 1e-3 else n
    s = sum(pi.values())
    pi = {a: v / s for a, v in pi.items()}
    z = sum(root.W.values()) / total if total else root.value
    return pi, z
