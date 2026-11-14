"""Side-symmetri: speglar ein stilling 180° og byter om kven som er kven.

Sáhkku-brettet er ein 60-stegs-syklus der: TOUR2[t] = rot(TOUR1[t]) — giera
sin syklus er menn sin syklus rotert 180° om midten (sjå engine._tour2_of).
Difor er spelet invariant under: roter brettet 180°, byt om eigarane, byt
side-i-trekket, og lat soldatane sine syklusposisjonar (p.at) stå. Det gjev
OSS gratis treningsdata: kvart sjølvspelparti vert òg spelt «frå andre sida»,
og nettet lærer Giera-stillingar like godt som Menn-stillingar (BLANKA.3
tapte 20-4 i eigen spegel som Giera — analysis/blanka3_overnight/FINDINGS.md
§2.4).

Verifisert av tests/test_side_flip.py: for kvar lovleg stilling i eit sett
parti, kvart lovleg trekk: etterfølgjar-FENen av (trekk i stilling) er
spegelen av etterfølgjar-FENen av (spegla trekk i spegla stilling).

MERK: augmenteringa skal skje i Sjølvspel-TIDA (selfplay), ikkje i treninga:
encode_state er dyr (capture_squares kallar trekk-generatoren), og spegel-
brettet må kodast med SAME koder for at nettet skal sjå ein gyldig stilling.
"""
from __future__ import annotations

import numpy as np

from . import engine as E
from . import features as FT

# rot[i] — 180°-rotasjonen i flatt brettindeks (kol 14-c, rad 2-r).
ROT = [E.idx_of(14 - E.pt(i)[0], 2 - E.pt(i)[1]) for i in range(E.N)]
# rot(22) = 22 (slottet): slottet ligg i midten og er invariant.
assert ROT[E.CASTLE] == E.CASTLE

# action-id-flip: berre SOLDAT-klassane ber eigaren i bokstaven (M=menn,
# K=giera). Dronninga (Q) og kongen (k) er same bokstav for båe eigarane —
# eigaren ligg i brettet, ikkje i trekk-id-en (features.move_id).
_LETTER_SWAP = {"M": "K", "K": "M", "Q": "Q", "k": "k"}


def flip_state(st: E.State) -> E.State:
    """Spegla stillinga: 180° rotasjon + eigarbyt + sidebyt (p.at står for
    soldatar — syklusposisjonen er den same, berre med andre eigar)."""
    s = st.clone()
    for p in s.pieces:
        if p.o is not None:
            p.o = 1 - p.o
        if p.t != "s":                 # konge/dronning: brettindeks
            p.at = ROT[p.at]
            p.t0 = ROT[p.t0]
    s.turn = 1 - s.turn
    s.kills = [s.kills[1], s.kills[0]]
    if s.winner is not None:
        s.winner = 1 - s.winner
    if s.dice_by is not None:
        s.dice_by = 1 - s.dice_by
    s.last_move = None                 # kodinga les ikkje han
    return s


def flip_action(aid: int) -> int:
    """action-id i den spegla stillinga (eit trekks identitet etter spegling)."""
    if aid >= FT.SPEC_BASE:
        return aid
    if aid >= FT.MOVE_BASE:
        j = aid - FT.MOVE_BASE
        letter = FT.CLASSES[j // 2025]
        frm = (j % 2025) // 45
        to = j % 45
        return (FT.MOVE_BASE + FT.CLASS_IDX[_LETTER_SWAP[letter]] * 2025
                + ROT[frm] * 45 + ROT[to])
    j = aid - FT.REL_BASE
    side = j // 90
    cls = (j % 90) // 45
    target = j % 45
    return FT.REL_BASE + (1 - side) * 90 + cls * 45 + ROT[target]


def flip_policy(pol: np.ndarray) -> np.ndarray:
    """policy-vektor i spegla stilling (dense 8282, same format)."""
    out = np.zeros_like(pol)
    nz = np.nonzero(pol)[0]
    for a in nz:
        out[flip_action(int(a))] = pol[a]
    return out


def flip_sample(st: E.State, pol: np.ndarray):
    """(planes, globals, policy, z) for den spegla stillinga — z står:
    verdi er målt frå sida-som-trekk sin POV, og ho er same POV etter bytet."""
    s = flip_state(st)
    pl, g = FT.encode_state(s)
    return pl, g, flip_policy(pol), s


def verify(spg_text: str, split_games, game_from_spg_any, to_fen,
           max_games: int = 3, every: int = 11) -> tuple:
    """Sanningstest: spegel-kommutativitet over ekte parti.

    For kvar stilling og kvart lovlege trekk:
      fen(apply(st, mv))  ==  spegel(fen(apply(st', mv')))
    Returnerer (n_stillingar, n_trekk, n_feil, fyrste feilmelding).
    """
    nst = nmv = bad = 0
    first = ""
    for gi, gt in enumerate(split_games(spg_text)[:max_games]):
        pairs = game_from_spg_any(gt, strict=False)
        samples = pairs[1] if isinstance(pairs, tuple) else pairs
        for st, _mv in samples[::every]:
            if st.winner is not None or st.phase != "move":
                continue
            sf = flip_state(st)
            nst += 1
            for aid, mv in FT.legal_ids(st):
                if aid >= FT.SPEC_BASE:
                    continue
                nmv += 1
                a = st.clone()
                E.apply_move(a, mv)
                b = sf.clone()
                # finn spegla trekk i spegla stilling
                fid = flip_action(aid)
                leg = dict(FT.legal_ids(sf))
                mvf = leg.get(fid)
                if mvf is None:
                    bad += 1
                    if not first:
                        first = (f"sp{gi}: spegla trekk {fid} er IKKJE lovleg "
                                 f"i spegla stilling (opphavleg {aid})")
                    continue
                E.apply_move(b, mvf)
                # konge-/dronninge-plassering vert flytta i apply_move; jamfør FEN
                if to_fen(flip_state(a)) != to_fen(b):
                    bad += 1
                    if not first:
                        first = (f"sp{gi}: FEN-skildnad etter spegling\n"
                                 f"  spegla(st): {to_fen(flip_state(a))}\n"
                                 f"  sf:         {to_fen(b)}")
    return nst, nmv, bad, first
