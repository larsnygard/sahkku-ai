"""SPG reader — parse .spg files (spec: SnartSahkku docs/SPG.md) into games.

Yields, per game: header dict + an ordered move list replayable on the
engine: [('move', fromSqName, toSqName), ('roll', d1, d2, d3), ('reroll', ...),
('open', first), ...] plus result. Used by the dataset builder and by
tools/spg_to_positions.py.
"""
from __future__ import annotations
import re
import sys
from . import engine as E
from .fen import sq_of_name

TAG_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
# §4.2: omkast-merket står etter sida og før kolon — «10. M (omkast #1): II _ X».
# Vi tek òg hale-forma, so eldre filer ikkje brest.
ROLL_RE = re.compile(r"^(\d+)\.\s+([MK])(?:\s*\(omkast\s*#(\d+)\))?:\s+(X|II|III|_)\s+(X|II|III|_)"
                     r"\s+(X|II|III|_)(?:\s*\(omkast\s*#(\d+)\))?\s*$")
OPEN_RE = re.compile(r"^1\.\s+open(?:#(\d+))?:\s*M\((X|II|III|_)\s+(X|II|III|_)\s+(X|II|III|_)\)\s+K\((X|II|III|_)\s+(X|II|III|_)\s+(X|II|III|_)\)\s*(?:->\s*([MK]))?\s*$")
MV_RE = re.compile(r"^(?:(II|III):\s*)?([MkKQ])([A-O][123])->([A-O][123])(\u00a7|x \[[^\]]*\])?\s*$")
REL_RE = re.compile(r"^X->([A-O][123])\s*$")
END_RE = re.compile(r"^(1-0|0-1|\*)\s*(?:\{(\w+)\})?\s*$")
XMARK_RE = re.compile(r"^X!\s*$")
# SPG 1.2-kommentarar (PGN-syntaks): {…} kan gå over fleire linjer og kan ikkje
# nestast — alt til fyrste } er kommentar. ; let til linjeslutt.
BRACE_RE = re.compile(r"\{([^{}]*)\}", re.S)


RESULT_LINE_RE = re.compile(r"^(1-0|0-1|\*)\s*\{[^{}]*\}\s*$")


def strip_comments(text: str):
    """Fjern kommentarar; returner (tekst utan kommentarar, liste med kommentarar).

    Kommentarar ber ingen spelestand: eit parti skal spelast likt med og utan
    dei. Vi tek vare på dei so verktøy kan syne eller skrive dei ut att.

    Særfall: resultatlinja «1-0 {dronning}» har eit {…} som er ÅRSAK, ikkje
    kommentar — END_RE les henne. Difor vert slike linjer haldne utanfor
    fjerninga og sette tilbake etterpå.
    """
    found = []
    PROT = "\x00SPGESULT\x00"
    protected, lines = [], []
    for raw in text.splitlines():
        if RESULT_LINE_RE.match(raw.strip()):
            protected.append(raw.strip())
            lines.append(PROT)
        else:
            lines.append(raw)

    def take(m):
        found.append(m.group(1).strip())
        return " "

    body = BRACE_RE.sub(take, "\n".join(lines))
    out = []
    for raw in body.splitlines():
        if raw.strip() == PROT:
            out.append(protected.pop(0) if protected else raw)
            continue
        if ";" in raw:
            cut, note = raw.split(";", 1)
            if note.strip():
                found.append(note.strip())
            raw = cut
        out.append(raw)
    return "\n".join(out), found


def parse_spg(text: str):
    """Return {'tags':…, 'events':[…], 'result':…, 'reason':…, 'bad':…}."""
    tags, events, result, reason, bad = {}, [], None, "", None
    text, comments = strip_comments(text)      # SPG 1.2: {…} og ; kommentarar
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = TAG_RE.match(line)
        if m:
            tags[m.group(1)] = m.group(2)
            continue
        if line == "SPG-V1":
            continue
        m = OPEN_RE.match(line)
        if m:
            if m.group(8):
                events.append(("open", 0 if m.group(8) == "M" else 1))
            continue
        m = ROLL_RE.match(line)
        if m:
            ev = ("reroll" if (m.group(3) or m.group(7)) else "roll",
                  (m.group(4), m.group(5), m.group(6)),
                  0 if m.group(2) == "M" else 1)
            events.append(ev)
            continue
        m = MV_RE.match(line)
        if m:
            events.append(("move", m.group(2), m.group(3), m.group(4),
                           m.group(5), m.group(1)))
            continue
        m = REL_RE.match(line)
        if m:
            events.append(("rel", m.group(1)))
            continue
        if XMARK_RE.match(line):
            continue
        m = END_RE.match(line)
        if m:
            result, reason = m.group(1), (m.group(2) or "")
            continue
        bad = line
    # [Variant "Nesseby"] — regelmodusen høyrer til partiet, ikkje til kallet
    tags.setdefault("nesseby", "1" if "nesseby" in str(tags.get("Variant", "")).lower() else "0")
    return {"tags": tags, "events": events, "result": result,
            "nesseby": "nesseby" in str(tags.get("Variant", "")).lower(),
            "reason": reason, "bad": bad, "comments": comments}


def game_from_spg(text: str, strict: bool = True, _force_nesseby: bool = False):
    """Replay an SPG game on the engine.

    Returns (state_final, samples). samples = list of
    (state_clone_before_move, legal_move_dict) for every executed move —
    supervised training data (state → expert move).
    Raises ValueError on irreconcilable notation when strict.
    """
    p = parse_spg(text)
    # Regelmodusen høyrer til partiet: eit Nesseby-parti som vert spela opp med
    # vanlege reglar hamnar i feil stilling (kongen byter eigar) og brest.
    NB = bool(_force_nesseby or p.get("nesseby"))
    st = E.State(nesseby=NB)
    samples = []
    pending = None  # current roll (faces, side)
    for ev in p["events"]:
        kind = ev[0]
        if kind == "open":
            st = E.State(first=ev[1], nesseby=NB)
            st.phase = "roll"
        elif kind in ("roll", "reroll"):
            faces, side = ev[1], ev[2]
            # The roll line defines the side to act: a player who casts and
            # has no moves never passes the turn via settle(). Trust the SPG.
            st.turn = side
            st.dice = list(faces)
            st.used = [d == "_" for d in faces]
            st.dice_by = st.turn
            st.phase = "move"
        elif kind == "rel":
            _apply_token(st, None, None, ev[1], samples, strict)
        elif kind == "move":
            _apply_token(st, ev[1], ev[2], ev[3], samples, strict,
                         want_v=2 if len(ev) > 5 and ev[5] == "II" else
                                  (3 if len(ev) > 5 and ev[5] == "III" else None))
    return st, samples


def game_from_spg_any(text: str, strict: bool = True):
    """Som game_from_spg, men prøver Nesseby om vanlege reglar ikkje går opp.

    Eldre filer manglar [Variant]-tag. Å spele eit Nesseby-parti med vanlege
    reglar bryt på eit tidspunkt (kongen har bytt eigar), so då prøver vi same
    partiet med Nesseby på — går det heilt opp, var det det.
    """
    try:
        return game_from_spg(text, strict=strict)
    except ValueError:
        if "nesseby" in str(parse_spg(text).get("nesseby", "")).lower() or parse_spg(text).get("nesseby"):
            raise
        return game_from_spg(text, strict=strict, _force_nesseby=True)


def _apply_token(st, piece_letter, from_name, to_name, samples, strict, want_v=None):
    """Match an SPG move token against the engine's legal pool and apply.

    SPG names squares absolutely (from Menn's side). A release token has no
    from-square; a regular token has letter + from + to. `want_v` (2/3) is
    the die value from an II:/III: prefix — required when several same-class
    pieces share a from-square and differ only by die.
    """
    pool = E.legal_moves(st, st.turn)

    def enc(mv):
        p = E.piece(st, mv["pid"])
        if p.t == "s":
            frm = E.soldier_at(p.o, p.at)
            to = E.soldier_at(p.o, p.at + (1 if mv["kind"] == "rel" else mv["v"]))
        elif mv.get("to") is not None:
            frm, to = p.at, mv["to"]
        else:
            ok, land, _ = E.path_for(p.at, mv["dir"], mv["v"])
            frm, to = p.at, land
        return E.sq_name(frm), E.sq_name(to), p

    def letter(p):
        if p.t == "q":
            return "Q"
        if p.t == "k":
            return "k"
        return "M" if p.o == 0 else "K"

    matches = []
    for mv in pool:
        f, t, p = enc(mv)
        if t != to_name:
            continue
        if from_name is not None and f != from_name:
            continue
        if piece_letter is not None and letter(p) != piece_letter:
            continue
        if want_v is not None and mv.get("v") != want_v:
            continue
        matches.append(mv)
    if not matches:
        if strict:
            raise ValueError(
                "SPG: ingen lovleg trekk mot «"
                + (f"{piece_letter}{from_name}->{to_name}" if from_name else f"X->{to_name}")
                + "» (pool: " + ",".join(f"{a}->{b}" for a, b, _ in map(enc, pool)) + ")")
        return
    mv = matches[0]
    samples.append((st.clone(), mv))
    E.apply_move(st, mv)
    E.settle(st)


def load_file(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return parse_spg(fh.read())


if __name__ == "__main__":
    tot = 0
    for f in sys.argv[1:]:
        g = load_file(f)
        tot += 1
        print(f"{f}: tags={len(g['tags'])} events={len(g['events'])} "
              f"result={g['result']} reason={g['reason']}"
              + (f" BAD LINE: {g['bad']}" if g["bad"] else ""))
    print(f"{tot} filer")
