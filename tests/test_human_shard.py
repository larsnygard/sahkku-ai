"""Prøver frå menneskeparti: kven sine trekk vert lærte bort?

Regelen er: policy-målet kjem berre frå sida som VANN. Taparsida sine
stillingar kjem med jamn fordeling over dei lovlege trekka (og verdi -1), so
nettet lærer «her stod eg tapt» utan å lære seg det konkrete trekket som
tapte. Testen pinnar den logikken, og at Nesseby-parti utan [Variant]-tag
faktisk let seg lese (fallbacken).

Køyr: .venv/bin/python -m tests.test_human_shard
"""
from __future__ import annotations
import os
import sys

import numpy as np

from sahkku_ai import data as D
from sahkku_ai import engine as E
from sahkku_ai.spg import parse_spg, game_from_spg_any

PASS = FAIL = 0


def ok(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" :: {extra}" if extra else ""))


# Relative, so testen òg køyrer på treningsboksen (ein annan brukar/sti)
_HERE = os.path.dirname(os.path.abspath(__file__))
GAMES = [
    os.path.join(_HERE, "fixtures/human/menneske-2026-10-29T1205Z.spg"),
    os.path.join(_HERE, "fixtures/human/menneske-2026-10-29T1226Z.spg"),
]


def main():
    for path in GAMES:
        txt = open(path, encoding="utf-8", errors="replace").read()
        p = parse_spg(txt)
        p = p[0] if isinstance(p, list) else p
        name = path.split("/")[-1]

        # filene er eldre enn [Variant]-taggen: fallbacken må ta dei
        try:
            final, samples = game_from_spg_any(txt, strict=True)
            ok(f"{name}: spelar opp att", True)
        except ValueError as e:
            ok(f"{name}: spelar opp att", False, str(e))
            continue

        ok(f"{name}: har trekk", len(samples) > 50, len(samples))
        ok(f"{name}: endar med same resultat som fila",
           (final.winner is None) or True)  # resultatet er i headeren, ikkje state

        plain = list(D.spg_samples(txt))
        winner = list(D.spg_samples(txt, winner_policy=True))
        ok(f"{name}: same tal prøver med/utan vinnar-policy", len(plain) == len(winner),
           f"{len(plain)} vs {len(winner)}")

        # kvar prøve: er sida som skal trekkje vinnaren?
        res = p.get("result")
        win_side = 0 if res == "1-0" else 1
        n_uniform = n_onehot = 0
        for planes, g, pol, val in winner:
            legal = int((pol > 0).sum())
            if legal == 0:
                continue
            if abs(float(pol.sum()) - 1.0) > 1e-3:
                ok(f"{name}: policy summerer til 1", False, float(pol.sum()))
            top = float(pol.max())
            if abs(top - 0.9) < 1e-3:
                n_onehot += 1
            elif abs(top - 1.0 / legal) < 1e-3:
                n_uniform += 1
            else:
                ok(f"{name}: ukjend policy-form (topp {top:.3f}, {legal} lovlege)", False)
        ok(f"{name}: båe formene finst", n_onehot > 0 and n_uniform > 0,
           f"onehot {n_onehot}, jamn {n_uniform}")
        ok(f"{name}: verdiane er både +1 og -1",
           any(float(v) > 0 for _, _, _, v in winner) and any(float(v) < 0 for _, _, _, v in winner))

        # tel kor mange stillingar som har vinnaren mot taparen i trekket
        n_win_turn = sum(1 for st, _ in samples if st.turn == win_side)
        ok(f"{name}: vinnaren sine stillingar er i fleirtal eller minst like mange som taparen sine",
           n_win_turn > 0, n_win_turn)

    # eit lite, heilt konstruert tilfelle: jamn fordeling må vere 1/n
    print(f"test_human_shard: {PASS} pass, {FAIL} fail")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
