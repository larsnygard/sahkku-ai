"""Straffar-motstandaren: menneskestrategien som slår NAIS.

Kvifor han finst: NAIS tapsstrategi er at ho aktiverer få soldatar og set dei
framme, og så vert den eine aktive brikka plukka. Ingen av sjølvspelpartia
straffa det — motparten var ho sjølv, som gjer den same feilen. Denne
motstandaren spelar slik eit menneske gjer det:

  1. Slå når slaget er TRYGT (brikka som slår kan ikkje takast att med ein gong).
     Dronning fyrst (vinst), så konge, så soldat.
  2. Aktiver soldatar opp til `hold_k` (standard 6) — mange nok til å ha noko å
     ta med, men ikkje heile garden ut i eit slagfelt.
  3. Elles: flytt den tryggaste brikka til eit felt motstandaren IKKJE kan slå
     på. Aldri fram i rekkjevidd for å «sjå kva som skjer».
  4. Kongen (om han er vår) er reserve: han kan alltid flyttast trygt, so vi
     slepp å kaste bort eit kast på ein risikabel soldat.

Han er deterministisk med vilje: same stilling gjev same trekk, so tap mot han
er eit stabilt signal, ikkje støy.

Køyr som lærar:  pipeline --train-opponent punisher
Test:           .venv/bin/python -m tests.test_punisher
"""
from __future__ import annotations

import random

from . import engine as E
from . import features as FT

HOLD_K = 6              # kor mange soldatar han aktiverer før han held opp


def _caps(st, side):
    """Felt `side` kan slå på NO (motoren si eiga sanning)."""
    return FT.capture_squares(st, side)


def _released(st, side):
    return [p for p in E.soldiers(st, side) if p.released and not p.captured]


def _value_of(st, mv):
    """Kor verdifullt er slaget? Dronning >> konge >> soldat; elles 0."""
    land = E.move_dest(st, mv)
    best = 0
    for q in E.occ(st, land):
        if q.captured or q.o == st.turn:
            continue
        best = max(best, {"q": 3, "k": 2, "s": 1}.get(q.t, 1))
    return best


def pick_action(st: E.State):
    """Vel handling for sida som er i trekket. None = ingen lovleg handling."""
    me = st.turn
    foe = 1 - me
    ids = FT.legal_ids(st)
    if not ids:
        return None

    # etter eit kast kan X-omkast vere eit val; straffaren kastar aldri om
    # (det er eit val utan verdi for han), men han MÅ kunne seie pass.
    moves = [(a, m) for a, m in ids if m.get("pid") is not None]
    if not moves:
        return {"kind": "pass"}

    foe_caps = _caps(st, foe)          # felt FIENDEN kan slå på (= farlege felt)

    # ---- 1. trygge slag ---------------------------------------------------
    baik = []
    for a, m in moves:
        v = _value_of(st, m)
        if v == 0:
            continue
        land = E.move_dest(st, m)
        # trygt = feltet fienden ikkje kan slå på, eller eit slag der det han
        # slår med er ei brikke dei ikkje kan ta att (dronning/konge)
        p = E.piece(st, m["pid"])
        safe = (land not in foe_caps) or p.t in ("q", "k")
        baik.append((v, 1 if safe else 0, a, m))
    if baik:
        baik.sort(key=lambda t: (t[1], t[0]), reverse=True)   # trygt fyrst, så verdi
        if baik[0][1] == 1:
            return baik[0][3]
        # berre risikable slag finst: ta det likevel om det er dronninga
        if baik[0][0] >= 2:
            return baik[0][3]

    # ---- 2. aktiver fleire soldatar --------------------------------------
    if len(_released(st, me)) < HOLD_K:
        rel = [(a, m) for a, m in moves if m.get("kind") == "rel"]
        if rel:
            # vel den løysinga som landar tryggast (heilt bak er best)
            rel.sort(key=lambda t: E.move_dest(st, t[1]) in foe_caps)
            return rel[0][1]

    # ---- 3. flytt trygt --------------------------------------------------
    safe_moves = [(a, m) for a, m in moves
                  if E.move_dest(st, m) not in foe_caps and m.get("kind") != "rel"]
    if safe_moves:
        # føretrekk soldatar (ikkje kongen unødig), og helst den som kjem
        # lengst fram utan å verte utsett
        def key(t):
            p = E.piece(st, t[1]["pid"])
            return (0 if p.t == "s" else 1, -E.cycle_index(me, E.move_dest(st, t[1]))
                    if p.t == "s" else 0)
        safe_moves.sort(key=key)
        return safe_moves[0][1]

    # ---- 4. konge som reserve, elles pass --------------------------------
    k = E.king(st)
    if k is not None and k.o == me and not k.captured:
        kmoves = [(a, m) for a, m in moves if E.piece(st, m["pid"]).t == "k"]
        if kmoves:
            return kmoves[0][1]
    return {"kind": "pass"}


def pick(st: E.State):
    """Speleklår variant: gjev eit trekk-objekt (mv) eller None."""
    a = pick_action(st)
    if a is None or a.get("kind") in ("pass", "reroll"):
        return None
    return a
