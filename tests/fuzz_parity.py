"""Fuzz parity: Python engine vs reference JS engine (vendor/engine.js).

Plays random-dice games in the python engine. At EVERY decision point it
records: fen, legal-pool (UCI tokens), chosen token, resulting fen. Then
ships the whole trajectory to tools/parity_probe.js which re-parses every
FEN, re-derives every pool, re-applies every move and compares. Any
divergence is a port bug.

Run:  python tests/fuzz_parity.py [games]
"""
import json
import random
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sahkku_ai import engine as E
from sahkku_ai.fen import to_fen


def token_for(st, mv):
    """Canonical parity identity: 'LETTER FROM TO' (see parity_probe.js).

    FEN keeps soldiers anonymous within a stack, so the letter class
    (M/K/Q/k) is the exact identity FEN preserves — square tokens alone
    are ambiguous when queen+king share a square, pids across FEN are not.
    """
    p = E.piece(st, mv["pid"])
    letter = "Q" if p.t == "q" else ("k" if p.t == "k" else ("M" if p.o == 0 else "K"))
    if p.t == "s":
        frm = E.soldier_at(p.o, p.at)
        to = E.soldier_at(p.o, p.at + (1 if mv["kind"] == "rel" else mv["v"]))
    elif mv.get("to") is not None:
        frm, to = p.at, mv["to"]
    else:
        ok, land, _ = E.path_for(p.at, mv["dir"], mv["v"])
        frm, to = p.at, land
    return f"{letter} {E.name_of(frm)} {E.name_of(to)}"


def pool_tokens(st):
    seen, out = set(), []
    for mv in E.legal_moves(st, st.turn):
        t = token_for(st, mv)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def play_game(rnd, max_plies=400):
    rand = rnd.random
    st = E.State(first=rnd.randint(0, 1))
    E.opening_roll(st, rand)
    steps = []  # (fen, pool, canonical_move_token, fen_after)
    for _ in range(max_plies):
        if st.winner is not None:
            break
        if st.phase == "roll":
            E.roll(st, rand)
            st.phase = "move"
        lm = E.legal_moves(st, st.turn)
        if not lm:
            if E.can_reroll_x(st):
                E.reroll_x(st, rand)
                st.phase = "move"
                continue
            E.pass_remaining(st)
            continue
        fen_before = to_fen(st)
        pool = pool_tokens(st)
        # S-UCI §4: når fleire brikker i ein stabel gjev same «frå-til», løyser
        # dommaren til den FRØMSTE (høgast syklusposisjon). Vel trekket i same
        # ånd, elles kan Python og JS flytte kvar si brikke i stabelen og
        # produsere ulik FEN av heilt lovlege grunnar.
        tok = pool[rnd.randrange(len(pool))]
        mv = max((m for m in lm if token_for(st, m) == tok),
                 key=lambda m: E.piece(st, m["pid"]).at)
        E.apply_move(st, mv)
        E.settle(st)
        steps.append((fen_before, pool, tok, to_fen(st)))
    return steps, st.winner


def main(games=40, seed=7):
    rnd = random.Random(seed)
    all_steps = []
    for g in range(games):
        steps, winner = play_game(rnd)
        all_steps.extend(steps)
    print(f"python: {games} parti, {len(all_steps)} trekkpunkt")
    probe = os.path.join(os.path.dirname(__file__), "..", "tools", "parity_probe.js")
    payload = {"positions": [{"fen": f, "mv": t} for f, pool, t, fa in all_steps]}
    proc = subprocess.run(
        ["node", probe], input=json.dumps(payload).encode(),
        capture_output=True, timeout=600)
    if proc.returncode != 0:
        print("JS-PROBE FEILA:", proc.stderr.decode()[:800])
        sys.exit(1)
    out = json.loads(proc.stdout.decode())
    fails = 0
    for (f, pool, t, fa), rec in zip(all_steps, out):
        if rec.get("error"):
            print(f"FAIL parse: {f} -> {rec['error']}")
            fails += 1
            continue
        if sorted(rec["pool"]) != sorted(pool):
            print(f"FAIL pool: {f}\n  py: {sorted(pool)}\n  js: {sorted(rec['pool'])}")
            fails += 1
        res = rec["result"]
        if not res or res.get("error"):
            print(f"FAIL apply: {f} mv {t}: {res}")
            fails += 1
            continue
        if res["fen"] != fa:
            print(f"FAIL fen mv {t}\n  FEN  : {f}\n  pool : {pool}\n  py   : {fa}\n  js   : {res['fen']}")
            fails += 1
    if fails:
        print(f"PARITET FEILA: {fails} av {len(all_steps)}")
        sys.exit(1)
    print(f"PARITET OK: {len(all_steps)} trekkpunkt, pool + FEN + apply identiske")


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    main(games, seed)
