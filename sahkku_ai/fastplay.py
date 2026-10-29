"""Vektorisert sjølvspel: K parti i lufta per arbeidar + CUDA-graph-inferens.

Kvifor: GPU-en er *latens*bunden på dette nettet. Eit forward kostar ~2,0 ms
same om batchen er 1, 8 eller 64 rad (≈15 kjerner à ~0,13 ms launch), og
produksjonstaket er ~1000 forward/s uansett kor mange arbeidarar som spør.
To grep gjer kvar forward mykje meir verdt:

  1. CUDA-graph med fast batch-storleik: 2,01 -> 0,49 ms for batch 8 (4,1×).
  2. K parti i lufta: alle utspelingane deira går i eitt og same forward per
     ply (batch = K × rollouts), so 8 avgjerder deler éin forward i staden for
     å stå i kø for 8.

Summen: ~8× fleire rad per forward og ~4× billegare forward. Regelarbeidet i
Python (≈30 % av tida) skalerer med K, so den totale vinsten er avgrensa av
det — målt i praksis 2-3× på heile løkka.

Datasettformatet er identisk med selfplay.py (data.save_shard), og reglane er
dei same: kvar avgjerd vert gjort med same π-mål og same vel av trekk.
"""
from __future__ import annotations
import math, os, random, time
import numpy as np
import torch

from . import engine as E
from . import features as FT
from .data import save_shard
from .model import Net
from .selfplay import MAX_PLIES, softmax_prior, sample_policy, apply_choice, dirichlet


class Predictor:
    """Fast-batch inferens. CUDA-graph når det går, elles vanleg forward."""

    def __init__(self, net, device, batch=64):
        self.net = net
        self.device = torch.device(device)
        self.B = int(batch)
        self.graph = None
        self.pol = self.val = None
        n_in = FT.N_PLANES * 45
        self.x = torch.zeros(self.B, n_in, device=self.device)
        self.g = torch.zeros(self.B, FT.N_GLOBAL, device=self.device)
        if self.device.type == "cuda":
            self._capture()

    def _capture(self):
        # varm opp (fyrste forward kompilerer/lastar kjerner)
        for _ in range(3):
            self.net(self.x, self.g)
        torch.cuda.synchronize()
        try:
            gr = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self.net(self.x, self.g)
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(gr):
                self.pol, self.val = self.net(self.x, self.g)
            torch.cuda.synchronize()
            self.graph = gr
        except Exception:                                    # noqa: BLE001
            self.graph = None                                # fell attende

    @torch.no_grad()
    def predict(self, states):
        """(policy_logits[n, A], value[n]) for ei liste stillingar."""
        n = len(states)
        out_p = np.empty((n, FT.N_ACTIONS), np.float32)
        out_v = np.empty(n, np.float32)
        for start in range(0, n, self.B):
            chunk = states[start:start + self.B]
            m = len(chunk)
            pl = np.empty((m, FT.N_PLANES * 45), np.float32)
            gl = np.empty((m, FT.N_GLOBAL), np.float32)
            for i, s in enumerate(chunk):
                planes, g = FT.encode_state(s)
                pl[i] = planes.reshape(-1)
                gl[i] = g
            self.x[:m].copy_(torch.from_numpy(pl))
            self.g[:m].copy_(torch.from_numpy(gl))
            if self.graph is not None:
                self.graph.replay()
                pol, val = self.pol, self.val
            else:
                pol, val = self.net(self.x, self.g)
            out_p[start:start + m] = pol[:m].detach().cpu().numpy()
            out_v[start:start + m] = val[:m].detach().cpu().numpy()
        return out_p, out_v


def _new_state(rng):
    st = E.State(first=rng.randint(0, 1))
    E.opening_roll(st, rng.random)
    return st


def _to_decision(st, rng):
    """Rull terningar / pass til stillinga har eit trekk eller er avgjort."""
    for _ in range(8):
        if st.winner is not None:
            return None
        if st.phase == "roll":
            E.roll(st, rng.random)
            st.phase = "move"
        ids = FT.legal_ids(st)
        if ids:
            return ids
        E.pass_remaining(st)
    return None


def play_batch(pred, cfg, rng, k_games, samples_out):
    """Spel K parti i lockstep; skriv prøver i same format som selfplay.py."""
    games = [_new_state(rng) for _ in range(k_games)]
    wins = [0, 0]
    R = cfg["rollouts"]

    while True:
        ctxs = []
        for i, st in enumerate(games):
            if st is None:
                continue
            ids = _to_decision(st, rng)
            if st.winner is not None:
                wins[st.winner] += 1
                games[i] = None
                continue
            if not ids:
                continue                     # vart passande, prøv neste runde
            ctxs.append({"i": i, "st": st, "ids": ids})
        if not ctxs:
            break

        # --- rot-policy for alle K avgjerdene i ÉIN forward -----------------
        pols, _ = pred.predict([c["st"] for c in ctxs])
        for c, pol in zip(ctxs, pols):
            legal = [a for a, _ in c["ids"]]
            c["legal"] = legal
            c["prior"] = softmax_prior(pol, legal)
            picks = [sample_policy(pol, legal, cfg["temp"], rng) for _ in range(R)]
            c["picks"] = picks
            c["children"] = []
            c["side0"] = []
            for a in picks:
                ch = c["st"].clone()
                apply_choice(ch, c["ids"], a, rng)
                c["children"].append(ch)
                c["side0"].append(c["st"].turn)   # kven som rørte seg i rota

        # --- alle utspelingar i lockstep -----------------------------------
        alive = [(ci, k) for ci, c in enumerate(ctxs) for k in range(len(c["children"]))]
        for _ in range(MAX_PLIES):
            if not alive:
                break
            batch, kinds = [], []
            for ci, k in alive:
                ch = ctxs[ci]["children"][k]
                if ch.winner is not None:
                    continue
                if ch.phase == "roll":
                    E.roll(ch, rng.random)
                    ch.phase = "move"
                ids = FT.legal_ids(ch)
                if not ids:
                    E.pass_remaining(ch)
                    continue
                batch.append(ch)
                kinds.append((ci, k, ids))
            if batch:
                bpols, _ = pred.predict(batch)
                for (ci, k, ids), pol in zip(kinds, bpols):
                    legal = [a for a, _ in ids]
                    aid = sample_policy(pol, legal, cfg["temp"], rng)
                    apply_choice(ctxs[ci]["children"][k], ids, aid, rng)
            alive = [(ci, k) for ci, k in alive if ctxs[ci]["children"][k].winner is None]

        # --- fullfør kvar avgjerd: π, z, vel trekk -------------------------
        for c in ctxs:
            # z må reknast frå ROTA SIN POV (sida som valde trekket) — same
            # konvensjon som evaluate_root i selfplay.py. Å rekne frå barnet
            # og so snu ein gong til gjev invertert forteikn (alle prøver
            # vart +1 i valideringa).
            root_side = c["st"].turn
            zs = []
            for ch in c["children"]:
                if ch.winner is None:
                    z = 0.0
                else:
                    z = 1.0 if ch.winner == root_side else -1.0
                zs.append(z)
            z_root = sum(zs) / max(1, len(zs))
            n = {a: 0 for a in c["legal"]}
            sz = {a: 0.0 for a in c["legal"]}
            for a, z in zip(c["picks"], zs):
                n[a] += 1
                sz[a] += z
            q = {a: (sz[a] / n[a] if n[a] else 0.0) for a in c["legal"]}
            N = max(1, len(c["picks"]))
            eps = cfg.get("prior_eps", 0.2)
            tau = max(1e-3, cfg.get("tau_q", 1.0))
            pi, tot = {}, 0.0
            for a, pr in zip(c["legal"], c["prior"]):
                w = (n[a] + eps * N * pr) * math.exp(q[a] / tau)
                pi[a] = w
                tot += w
            for a in c["legal"]:
                pi[a] /= max(tot, 1e-12)
            if cfg["noise_alpha"] > 0:
                d = dirichlet(rng, cfg["noise_alpha"], len(c["legal"]))
                for a, dv in zip(c["legal"], d):
                    pi[a] = (1 - cfg["noise_eps"]) * pi[a] + cfg["noise_eps"] * dv

            planes, g = FT.encode_state(c["st"])
            pol_t = np.zeros(FT.N_ACTIONS, np.float32)
            for a in c["legal"]:
                pol_t[a] = pi[a]
            z_out = np.float32(z_root)      # z_root er alt eit middel
            samples_out.append((planes.reshape(-1), g, pol_t, z_out))

            best = max(c["legal"], key=lambda a: pi[a])
            apply_choice(c["st"], c["ids"], best, rng)
            # avgjerda er teken; neste runde tek neste terning i same tur
            if c["st"].winner is None:
                E.settle(c["st"])
    return wins


def worker_batch(args):
    wid, args_json = args
    import json
    cfg = json.loads(args_json)
    torch.set_grad_enabled(False)
    device = torch.device(cfg["device"])
    net = Net.load(cfg["net"], map_location="cpu").to(device).eval()
    pred = Predictor(net, device, cfg.get("batch", 64))
    rng = random.Random(cfg["seed"] * 1000 + wid)
    samples, wins = [], [0, 0]
    t0 = time.time()
    per = cfg["games_per_worker"]
    k = max(1, min(cfg.get("k_games", 4), per))
    done = 0
    while done < per:
        kk = min(k, per - done)
        w = play_batch(pred, cfg, rng, kk, samples)
        wins[0] += w[0]
        wins[1] += w[1]
        done += kk
        if done % 10 == 0 or done == per:
            el = max(1e-9, time.time() - t0)
            print(f"[w{wid}] {done}/{per} games, {len(samples)} samples, {el:.0f}s "
                  f"({done / el * 60:.1f} games/min, graph={'yes' if pred.graph else 'no'})",
                  flush=True)
    out = os.path.join(cfg["outdir"], f"shard_{wid:03d}.npz")
    save_shard(samples, out)
    print(f"[w{wid}] done: {per} games, {len(samples)} samples -> {out} "
          f"(Menn wins {wins[0]}, Giera wins {wins[1]})", flush=True)
    return len(samples)
