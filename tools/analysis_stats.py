#!/usr/bin/env python3
"""Analyse SPG-fil og skriv oppførselsstatistikk per parti som JSONL.

Køyr: .venv/bin/python tools/analysis_stats.py <fil.spg> <ut.jsonl>

Per parti (éin JSONL-rad):
  result, plies, winreason, names, per-side-mål:
  * released-gjennomsnitt (frå ply 12)
  * exposed-gjennomsnitt (utløyste brikker på felt fienden kan slå på NO)
  * tryggleik: del av eigne trekk som landa på eit felt fienden kunne slå på
  * tal på slag tekne / gjevne (soldat/dronning/konge)
  * opningsfase: ply for første soldatløysing, ply for dronningløysing
  * dronninga hengd: første ply der taparen si dronning stod slagbar etter
    eige trekk (+ ply til ho faktisk vart teken)
  * konge: gjennomsnittleg rad (1 heim … 3 framme) for eigar-sida
  * pass tal, omkast tal
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sahkku_ai import engine as E
from sahkku_ai import features as FT
from sahkku_ai.spg import parse_spg, game_from_spg_any

_CAP_CACHE: dict = {}


def released_count(st, side):
    return sum(1 for p in E.soldiers(st, side) if p.released and not p.captured)


def exposed_count(st, side):
    caps = FT.capture_squares(st, 1 - side)   # felt FIENDEN kan slå på NO
    n = 0
    for p in E.soldiers(st, side):
        if p.released and not p.captured and E.soldier_at(side, p.at) in caps:
            n += 1
    return n


def kq_rank(st, side):
    k = E.king(st)
    if k is None or k.o != side:
        return None
    return E.pt(k.at)[1] + 1


def split_games(text: str):
    """Del SPG-tekst i parti (resultatlinje avsluttar kvart).

    Resultatlinjer: '1-0', '0-1', '*' — med valfri '{årsak}' (også utforma
    '* {vakt}'). END_RE krev klammer, så ei naken '*' er òg slutt.
    """
    import re
    end_re = re.compile(r"^(1-0|0-1|\*)(\s*\{[^{}]*\})?\s*$")
    games, cur = [], []
    for line in text.splitlines():
        cur.append(line)
        if end_re.match(line.strip()):
            games.append("\n".join(cur))
            cur = []
    if len([l for l in cur if l.strip()]) > 1:
        games.append("\n".join(cur))
    return games


def analyse(text: str):
    rows = []
    for gt in split_games(text):
        head = parse_spg(gt)
        head = head[0] if isinstance(head, list) else head
        pairs = game_from_spg_any(gt, strict=False)
        if not isinstance(pairs, tuple) and not isinstance(pairs, list):
            continue
        samples = pairs[1] if isinstance(pairs, tuple) else pairs
        final = pairs[0] if isinstance(pairs, tuple) else None
        result = head.get("result", "?")
        winner = 0 if result == "1-0" else (1 if result == "0-1" else None)
        names = {0: head.get("menn", "Menn"), 1: head.get("koner", "Koner")}

        rel_acc = [0.0, 0.0]
        rel_n = 0
        exp_acc = [0.0, 0.0]
        kr_rank = [[], []]
        land_total = [0, 0]      # eige trekk som landa på fiendens slagfelt
        land_safe = [0, 0]
        took = [{}, {}]          # {'s':n,'q':n,'k':n}
        lost = [{}, {}]
        first_rel = [None, None]
        q_rel_ply = [None, None]
        passes = [0, 0]
        rerolls = [0, 0]
        hang = [None, None]      # (ply, sq, taken_delta)

        n = len(samples)
        for i in range(n):
            st, mv = samples[i]
            after = samples[i + 1][0] if i + 1 < n else final
            if st.winner is None and i >= 12:
                rel_acc[0] += released_count(st, 0)
                rel_acc[1] += released_count(st, 1)
                exp_acc[0] += exposed_count(st, 0)
                exp_acc[1] += exposed_count(st, 1)
                rel_n += 1
                for s in (0, 1):
                    r = kq_rank(st, s)
                    if r is not None:
                        kr_rank[s].append(r)
            if not mv:
                continue
            side = st.turn
            # pass/reroll: mv har pid None
            if mv.get("pid") is None:
                if mv.get("kind") == "pass":
                    passes[side] += 1
                elif mv.get("kind") == "reroll":
                    rerolls[side] += 1
                continue
            p = E.piece(st, mv["pid"])
            land = E.move_dest(st, mv)
            if mv.get("kind") == "rel":
                if p.t == "s" and first_rel[side] is None:
                    first_rel[side] = i + 1
                if p.t == "q" and q_rel_ply[side] is None:
                    q_rel_ply[side] = i + 1
            if side < 2:
                foe_caps = FT.capture_squares(after, 1 - side) if after is not None else set()
                land_total[side] += 1
                if land in foe_caps:
                    land_safe[side] += 0
                else:
                    land_safe[side] += 1
            victims = [q for q in E.occ(st, land) if not q.captured and q.o is not None and q.o != side]
            for q in victims:
                took[side][q.t] = took[side].get(q.t, 0) + 1
                if q.o is not None:
                    lost[q.o][q.t] = lost[q.o].get(q.t, 0) + 1

        # dronning-hengjing for taparen
        loser = None if winner is None else 1 - winner
        if loser is not None:
            for i in range(n):
                st, mv = samples[i]
                if st.turn != loser or st.winner is not None:
                    continue
                after = samples[i + 1][0] if i + 1 < n else final
                q = E.queen(after, loser) if after is not None else None
                if q is None or q.captured:
                    continue
                if after is not None and q.at in FT.capture_squares(after, winner):
                    taken = None
                    for j in range(i + 1, n):
                        sj = samples[j + 1][0] if j + 1 < n else final
                        qj = E.queen(sj, loser) if sj is not None else None
                        if qj is None or qj.captured:
                            taken = j - i
                            break
                    hang[loser] = (i + 1, E.sq_name(q.at), taken)
                    break

        row = {
            "result": result,
            "names": names,
            "plies": n,
            "winreason": head.get("result", ""),
            "avg_released": [round(rel_acc[s] / rel_n, 2) if rel_n else None for s in (0, 1)],
            "avg_exposed": [round(exp_acc[s] / rel_n, 2) if rel_n else None for s in (0, 1)],
            "safe_land_frac": [round(land_safe[s] / land_total[s], 3) if land_total[s] else None
                               for s in (0, 1)],   # del av eige trekk som landa utanfor fiendens slagfelt
            "took": took, "lost": lost,
            "first_soldier_release": first_rel,
            "queen_release_ply": q_rel_ply,
            "queen_hang": hang,
            "avg_king_rank": [round(sum(kr_rank[s]) / len(kr_rank[s]), 2) if kr_rank[s] else None
                              for s in (0, 1)],
            "passes": passes, "rerolls": rerolls,
        }
        rows.append(row)
    return rows


def main():
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src, encoding="utf-8", errors="replace").read()
    rows = analyse(text)
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} games -> {dst}")


if __name__ == "__main__":
    main()
