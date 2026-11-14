#!/usr/bin/env python3
"""Aggregér per-parti JSONL (frå tools/analysis_stats.py) til ein samanlikning.

Køyr: .venv/bin/python tools/analysis_agg.py <fil.jsonl> <A-mappa> [fil2.jsonl A-mappa ...]

A-mappa:
  'menndom'  A spelar Menn i parti 0,2,4… og Giera i 1,3,5… (eval_match-regelen)
  'mirror'   begge sider er same motor — rapportér per side, ikkje per motor
  'classicm' CPU-mirror: side0=Menn-CPU, side1=Giera-CPU

Skriv tabell per fil:
  * resultatfordeling (vinst/tap per side)
  * snitt per side: released, exposed, safe-land, takne/teke brikker,
    ply for første soldat-/dronningløysing, hengt-dronning-dekning, konge-rad
"""
from __future__ import annotations
import json
import sys


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def fmt(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "  -  "


def agg_side(rows, pick):
    """pick(row, gi) -> 0/1 sida me tel på.»"""
    n = wins = 0
    acc = dict(rel=0.0, exp=0.0, safe=0.0, took_s=0, took_q=0, took_k=0,
               lost_s=0, lost_q=0, lost_k=0, first_rel=[], q_rel=[],
               hang_ply=[], hang_taken=[], hang_share=0, krank=[],
               passes=0, rerolls=0, plies=0, games=0)
    for gi, r in enumerate(rows):
        s = pick(r, gi)
        if s is None:
            continue
        acc["games"] += 1
        acc["plies"] += r["plies"]
        w = 0 if r["result"] == "1-0" else (1 if r["result"] == "0-1" else None)
        if w is not None and w == s:
            acc["_wins"] = acc.get("_wins", 0) + 1
        rel = r["avg_released"][s]
        exp = r["avg_exposed"][s]
        sf = r["safe_land_frac"][s]
        if rel is not None:
            acc["rel"] += rel
        if exp is not None:
            acc["exp"] += exp
        if sf is not None:
            acc["safe"] += sf
        t, l = r["took"][s], r["lost"][s]
        acc["took_s"] += t.get("s", 0); acc["took_q"] += t.get("q", 0); acc["took_k"] += t.get("k", 0)
        acc["lost_s"] += l.get("s", 0); acc["lost_q"] += l.get("q", 0); acc["lost_k"] += l.get("k", 0)
        if r["first_soldier_release"][s] is not None:
            acc["first_rel"].append(r["first_soldier_release"][s])
        if r["queen_release_ply"][s] is not None:
            acc["q_rel"].append(r["queen_release_ply"][s])
        h = r["queen_hang"][s]
        if h is not None:
            acc["hang_share"] += 1
            acc["hang_ply"].append(h[0])
            if h[2] is not None:
                acc["hang_taken"].append(h[2])
        else:
            acc.setdefault("_nohang", 0); acc["_nohang"] += 1
        if r["avg_king_rank"][s] is not None:
            acc["krank"].append(r["avg_king_rank"][s])
        acc["passes"] += r["passes"][s]
        acc["rerolls"] += r["rerolls"][s]
    g = max(1, acc["games"])
    mean = lambda k: acc[k] / g
    avg = lambda xs: sum(xs) / len(xs) if xs else None
    return {
        "games": acc["games"],
        "wins": acc.get("_wins", 0),
        "plies": fmt(mean("plies"), 0),
        "released": fmt(mean("rel")),
        "exposed": fmt(mean("exp")),
        "safe_land%": fmt(100 * mean("safe"), 1),
        "took s/q/k": f"{fmt(mean('took_s'),1)}/{fmt(mean('took_q'),1)}/{fmt(mean('took_k'),1)}",
        "lost s/q/k": f"{fmt(mean('lost_s'),1)}/{fmt(mean('lost_q'),1)}/{fmt(mean('lost_k'),1)}",
        "first rel ply": fmt(avg(acc["first_rel"]), 1),
        "queen rel ply": fmt(avg(acc["q_rel"]), 1),
        "hung queen": f"{acc['hang_share']}/{acc['games']}",
        "hang->taken": fmt(avg(acc["hang_taken"]), 1),
        "king rank": fmt(avg(acc["krank"])),
        "pass/reroll g": f"{fmt(mean('passes'),1)}/{fmt(mean('rerolls'),1)}",
    }


def main():
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        path, mode = args[i], args[i + 1]
        rows = load(path)
        print(f"\n===== {path}  ({len(rows)} games) =====")
        if mode == "menndom":
            # eval_match: game g (0-idx): A=Menn (side0) om g partall
            for label, pick in (
                ("A-laga", lambda r, gi: 0 if gi % 2 == 0 else 1),
                ("B-laga", lambda r, gi: 1 if gi % 2 == 0 else 0),
            ):
                st = agg_side(rows, pick)
                print(f"{label}: " + "  ".join(f"{k}={v}" for k, v in st.items()))
        else:
            for label, pick in (("Menn (side 0)", lambda r, gi: 0),
                                ("Giera (side 1)", lambda r, gi: 1)):
                st = agg_side(rows, pick)
                print(f"{label}: " + "  ".join(f"{k}={v}" for k, v in st.items()))


if __name__ == "__main__":
    main()
