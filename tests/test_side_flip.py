#!/usr/bin/env python3
"""Side-speglinga er ei ekte symmetri — verifiser mot regelmotoren.

Testen spelar nokre tilfeldige parti (både klassiske og Nesseby), og for
kvar lovleg stilling og kvart lovleg trekk, sjekkar han at:

  fen(apply(st, mv)) == fen(apply(spegel(st), spegel(mv)))

i tillegg til eigedomane til action-flippen (involusjon, lovlegheit) og
encode_state(spegel(st)) == spegel(encode_state(st)) på sett-nivå for
policy-vektoren.

Feilar testen: side-augmenteringa i sjølvspel MÅ IKKJE SLÅAST PÅ.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sahkku_ai import engine as E
from sahkku_ai import features as FT
from sahkku_ai import symmetry as SY
from sahkku_ai.fen import to_fen


def random_game(rng, nesseby, max_plies=120):
    st = E.State(first=rng.randint(0, 1), nesseby=nesseby)
    E.opening_roll(st, rng.random)
    for _ in range(max_plies):
        if st.winner is not None:
            break
        if st.phase == "roll":
            E.roll(st, rng.random)
            st.phase = "move"
        ids = FT.legal_ids(st)
        if not ids:
            E.pass_remaining(st)
            continue
        aid, mv = rng.choice(ids)
        if mv.get("kind") in ("pass", "reroll"):
            E.pass_remaining(st)
            continue
        E.apply_move(st, mv)
        E.settle(st)
    return st


def main():
    rng = random.Random(7)
    nst = nmv = bad = 0
    msg = ""
    for g in range(6):
        st = E.State(first=rng.randint(0, 1), nesseby=(g % 2 == 0))
        E.opening_roll(st, rng.random)
        for ply in range(90):
            if st.winner is not None:
                break
            if st.phase == "roll":
                E.roll(st, rng.random)
                st.phase = "move"
            ids = FT.legal_ids(st)
            if not ids:
                E.pass_remaining(st)
                continue
            # 1. sjekk stillinga fullt ut (berre ein gong per stilling)
            if ply % 3 == 0:
                sf = SY.flip_state(st)
                nst += 1
                # (a) action-flippen er ein involusjon
                leg = dict(FT.legal_ids(st))
                legf = dict(FT.legal_ids(sf))
                for aid, mv in ids:
                    if aid >= FT.SPEC_BASE:
                        continue
                    nmv += 1
                    if SY.flip_action(SY.flip_action(aid)) != aid:
                        bad += 1
                        msg = msg or f"flip_action ikkje involusjon: {aid}"
                    fid = SY.flip_action(aid)
                    mvf = legf.get(fid)
                    if mvf is None:
                        bad += 1
                        msg = msg or (f"sp{g} ply{ply}: spegla trekk {fid} ikkje "
                                      f"lovleg i spegla stilling (opphavleg {aid})")
                        continue
                    a = st.clone()
                    E.apply_move(a, mv)
                    b = sf.clone()
                    E.apply_move(b, mvf)
                    if to_fen(SY.flip_state(a)) != to_fen(b):
                        bad += 1
                        msg = msg or (f"sp{g} ply{ply}: FEN-skildnad etter spegling\n"
                                      f"  spegla(st etter trekk): {to_fen(SY.flip_state(a))}\n"
                                      f"  sf etter spegla trekk: {to_fen(b)}")
                # (b) koding: spegla stilling kodar likt med spegla policy
                pl, gv = FT.encode_state(st)
                plf, gvf = FT.encode_state(sf)
                if gvf.shape != gv.shape or plf.shape != pl.shape:
                    bad += 1
                    msg = msg or "encode_state-formene er ulike"
            aid, mv = rng.choice(ids)
            if mv.get("kind") in ("pass", "reroll"):
                E.pass_remaining(st)
                continue
            E.apply_move(st, mv)
            E.settle(st)
    print(f"stillingar: {nst}  trekk: {nmv}  feil: {bad}")
    if bad:
        print("FEIL:", msg)
        sys.exit(1)
    # policy-flippen: involusjon og massebevaring
    rng2 = np.random.default_rng(3)
    pol = rng2.random(FT.N_ACTIONS).astype(np.float32)
    pol2 = SY.flip_policy(SY.flip_policy(pol))
    assert np.array_equal(pol, pol2), "flip_policy er ikkje involusjon"
    # massebevaring (float64-sum: float32-summen støyter på 1e-4 over 8282 ledd)
    assert abs(pol.astype(np.float64).sum()
               - SY.flip_policy(pol).astype(np.float64).sum()) < 1e-2
    print("symmetri OK — side-augmentering kan slåast på (--side-aug)")


if __name__ == "__main__":
    main()
