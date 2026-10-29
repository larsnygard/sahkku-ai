"""Verifiser ein SPG-fil: kan ho lesast att og gjev ho same sluttstilling?

Går gjennom den verkelege parsaren i repoet (sahkku_ai/spg.py), spelar
partiet opp att trekk for trekk (streng modus: ukjende trekk feilar), og
samanliknar slutt-FEN med stillinga partiet faktisk enda i.

    .venv/bin/python tools/verify_spg.py <fil.spg> <fasit-fen|-> [json]
"""
import json
import sys

sys.path.insert(0, "/home/lars/src/sahkku-ai")
from sahkku_ai import spg as SPG
from sahkku_ai import fen as FEN


def main():
    path = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else "-"
    text = open(path).read()
    p = SPG.parse_spg(text)
    print(f"tags: {p['tags']}")
    print(f"result: {p['result']}  reason: {p['reason']}  events: {len(p['events'])}")
    st, samples = SPG.game_from_spg(text, strict=True)
    got = FEN.to_fen(st)
    print(f"moves replayed: {len(samples)}")
    print(f"slutt-FEN: {got}")
    if want not in ("-", ""):
        print("SAMANLIKN MED FASIT:", "LIK" if got.strip() == want.strip() else "ULIK")
        if got.strip() != want.strip():
            print("  fasit:", want)
            sys.exit(1)
    # partiet skal vere uavgjort på taket: ingen vinnar
    print("vinnar:", st.winner, "| fase:", st.phase)


if __name__ == "__main__":
    main()
