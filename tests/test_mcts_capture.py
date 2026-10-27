"""Søket må ta ein vinnande kapsel, sjølv med eit heilt nøytralt nett.

Bakgrunn (Lars 2026-10-27): i eit parti vart det klaga på at AI-en ikkje tok
konge-/dronningkapslar som var rett fram. Årsaka var i MCTS, ikkje i reglane:

  * eit AVGJORT løv (til dømes eit trekk som tek dronninga og vinn) fekk
    verdien sin sett, men vart aldri ført oppover — så forelderen sin N/W
    stod på 0 og trekket vart aldri kreditert;
  * eit ferdig born vart hoppa over i seleksjonen, so søket gjekk attende
    til eit roleg trekk;
  * resten av sim-budsjettet rann ut i inkje, og π (vitjingsbasert) peika
    på feil trekk.

Testen brukar eit stub-nett som gjev heilt Flat prior og verdi 0. Då kan
berre terminal-handteringa gjere at slaget vert valt — så testen isolerer
søket, ikkje nettet.

Køyr:  .venv/bin/python -m tests.test_mcts_capture
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sahkku_ai import engine as E
from sahkku_ai import features as FT
from sahkku_ai import mcts as MCTS

OK = []


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    OK.append(msg)


class FlatStub:
    """Nøytralt «nett»: lik prior på alt, verdi 0 (veit ingenting)."""

    def predict(self, states):
        n = len(states)
        pols = np.full((n, FT.N_ACTIONS), 1.0 / FT.N_ACTIONS, np.float32)
        vals = np.zeros(n, np.float32)
        return pols, vals


def rig(v=3, soldiers=3):
    """Side 0 si dronning kan ta side 1 si dronning; soldatar på båe sider
    so stillinga IKKJE alt er avgjort på utryddelse."""
    st = E.State(first=0)
    st.queen_mode = True
    st.phase = "move"
    st.turn = 0
    for side in (0, 1):
        for i, p in enumerate(E.soldiers(st, side)):
            p.captured = i >= soldiers
            if i < soldiers:
                p.released = False
                p.at = i
    q0, q1 = E.queen(st, 0), E.queen(st, 1)
    q0.captured, q0.released = False, True
    q1.captured, q1.released = False, True
    face = {1: "X", 2: "II", 3: "III"}[v]
    for a in range(45):
        for b in range(45):
            if a == b:
                continue
            q0.at, q1.at = a, b
            st.dice = [face, "_", "_"]
            st.used = [False, True, True]
            for mv in E.legal_moves(st, 0):
                if mv["pid"] != q0.id:
                    continue
                if E.move_dest(st, mv) == b and any(q is q1 for q in E.occ(st, b)):
                    return st, mv
    return None, None


def main():
    cfg = {"sims": 25, "c_puct": 1.6, "leaf_batch": 8, "temp": 1.0,
           "noise_alpha": 0.0, "noise_eps": 0.0, "rollouts": 8, "prior_eps": 0.2,
           "tau_q": 1.0, "batch": 8}
    stub = FlatStub()

    for v in (1, 2, 3):
        st, cap = rig(v)
        check(st is not None, f"stilling med dronningkapsel (terning {v}) finst")
        ids = FT.legal_ids(st)
        check(any(m["pid"] == cap["pid"] and E.move_dest(st, m) == E.move_dest(st, cap)
                  for _, m in ids), f"v{v}: kapselen er med i handlingmengda")
        # stillinga må ikkje alt vere vunnen (elles vinn kvart trekk)
        check(st.winner is None and len(E.alive_soldiers(st, 1)) > 0,
              f"v{v}: stillinga er ikkje alt avgjort")
        pi, z = MCTS.search(stub, st, dict(cfg), random.Random(7), root_noise=False)
        check(bool(pi), f"v{v}: søket gav ei fordeling")
        pick = max(pi, key=pi.get)
        mv = next(m for a, m in ids if a == pick)
        took = (mv["pid"] == cap["pid"] and E.move_dest(st, mv) == E.move_dest(st, cap))
        check(took, f"v{v}: MCTS tok dronningkapselen (π={pi[pick]:.2f}), ikkje eit roleg trekk")
        check(pi[pick] > 0.5, f"v{v}: kapselen fekk klårt fleirtal av vitjingane (π={pi[pick]:.2f})")

    print(f"MCTS-KAPSELTEST OK: {len(OK)} sjekkar")
    for m in OK:
        print("  ok", m)


if __name__ == "__main__":
    main()
