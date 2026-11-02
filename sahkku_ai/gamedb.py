"""Indexert speltdatabase for sáhkku-parti — same metode som SCID brukar for
sjakkbrett-parti: to filar med felles stamnamn.

  <prefix>.sci   indeks: fast breidde per parti (magic-header + éi fast
                 post per parti: offset i datafila, komprimert/rå lengd,
                 resultat, terningvariant, plies, dato, tag-strengar).
                 SCID si .sci — kan lesast utan å røre spelteksten.
  <prefix>.sgt   speltekst: SPG-tekstane, zlib-komprimerte, lagde étt og
                 étt etter kvart som ein berre appendar. SCID si .sg4.

Kvar post peikar på (offset, komprimert lengd) i .sgt-fila; partiet er
dekomprimert SPG-tekst, byte-for-byte identisk med det vi importerte.
Søking/filtrering skjer i indekspostane åleine (fast struktur = O(1)-opp-
slaget), og heile databasen treng ikkje lesast inn.

Bruk som modul:
    from sahkku_ai import gamedb
    db = gamedb.open_db("runs/games")          # opnar eller lagar
    db.add(spg_text, verify=True)              # returnerer game-id
    db.list(result="1-0", nesseby=True)       # indeks-postar
    db.get(gid)                                # SPG-tekst
    db.close()

Felt verdar merknad:
  * `plies` er halvtrekk (same teljing som eval_match/rapportane).
  * `result`: 0=oppgjer 1=Menn vann 2=Giera vann 3=uavgjort.
  * `date`: YYYYMMDD som heiltal (SCID-variant), 0 = ukjend.
"""
from __future__ import annotations
import os
import re
import struct
import zlib

MAGIC = b"SAHKGDB1"
VERSION = 1
# indekspost: id, offset, clen, rlen, result, nesseby, plies, date, term, res
# + event(40) white(24) black(24) termination(12) variant(12)
RECFMT = "<I Q I I B B H I H H 40s 24s 24s 12s 12s"
RECSZ = struct.calcsize(RECFMT)
HDRFMT = "<8s I I I I"          # magic, version, n_games, rec_size, flags
HDRSZ = struct.calcsize(HDRFMT)

RESULT_CODE = {"1-0": 1, "0-1": 2, "1/2-1/2": 3, "*": 0, "": 0}
RESULT_STR = {v: k for k, v in RESULT_CODE.items()}
RES_MENN, RES_GIERA, RES_DRAW = 1, 2, 3

TAG_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')


def _tags_of(spg_text: str) -> dict:
    out = {}
    for line in spg_text.splitlines():
        m = TAG_RE.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _ply_count(spg_text: str) -> int:
    # tel halvtrekk: trekklene har format «N. MFR->T» (éi linje per halvtrekk
    # i SPG-en; opnings-/passlinjer tel ikkje)
    n = 0
    for line in spg_text.splitlines():
        s = line.strip()
        m = re.match(r"^\d+\.\s+[A-Zk§]", s)
        if m:
            n += 1
    return n


def _date_int(tags: dict) -> int:
    d = tags.get("Date", "")
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", d)
    return int(m.group(1)) * 10000 + int(m.group(2)) * 100 + int(m.group(3)) if m else 0


def _pack(s: str, width: int) -> bytes:
    b = (s or "").encode("utf-8", "replace")[:width]
    return b.ljust(width, b"\x00")


def _unpack(b: bytes) -> str:
    return b.split(b"\x00", 1)[0].decode("utf-8", "replace")


class Game:
    """Ein indekspost (lesbar utan datafila)."""
    __slots__ = ("gid", "offset", "clen", "rlen", "result", "nesseby",
                 "plies", "date", "event", "white", "black",
                 "termination", "variant")

    def __init__(self, rec: bytes, gid: int):
        (self.gid, self.offset, self.clen, self.rlen, self.result,
         self.nesseby, self.plies, self.date, _term, _res,
         event, white, black, termination, variant) = struct.unpack(RECFMT, rec)
        self.event = _unpack(event)
        self.white = _unpack(white)
        self.black = _unpack(black)
        self.termination = _unpack(termination)
        self.variant = _unpack(variant)

    @property
    def result_str(self) -> str:
        return RESULT_STR.get(self.result, "*")

    def as_dict(self) -> dict:
        return {"gid": self.gid, "result": self.result_str,
                "nesseby": bool(self.nesseby), "plies": self.plies,
                "date": self.date, "event": self.event, "white": self.white,
                "black": self.black, "termination": self.termination,
                "variant": self.variant, "offset": self.offset,
                "clen": self.clen, "rlen": self.rlen}

    def __repr__(self):
        return (f"Game({self.gid}: {self.white}–{self.black} "
                f"{self.result_str} {self.plies}p)")


class GameDB:
    """SCID-methoden: .sci-indeks + .sgt-comprimert speltekst, éin opna fil."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.sci = prefix + ".sci"
        self.sgt = prefix + ".sgt"
        self._sgtf = None
        if os.path.exists(self.sci):
            with open(self.sci, "rb") as f:
                hdr = f.read(HDRSZ)
                magic, ver, n, rsz, _fl = struct.unpack(HDRFMT, hdr)
                if magic != MAGIC:
                    raise SystemExit(f"{self.sci}: not a sáhkku game DB (magic {magic!r})")
                if ver != VERSION or rsz != RECSZ:
                    raise SystemExit(f"{self.sci}: unsupported format v{ver} rec {rsz}")
                self.count = n
                self._records = f.read()
                if len(self._records) != n * RECSZ:
                    raise SystemExit(f"{self.sci}: index size {len(self._records)} "
                                     f"!= {n} * {RECSZ}")
                # datafilas lengd må stemme med indeksens sluttoffset
                sz = os.path.getsize(self.sgt) if os.path.exists(self.sgt) else 0
                if n:
                    last = struct.unpack(RECFMT, self._records[-RECSZ:])
                    end = last[1] + last[2]
                    if sz < end:
                        raise SystemExit(f"{self.sgt}: truncated "
                                         f"({sz} bytes, index expects {end})")
        else:
            self.count = 0
            self._records = b""
            with open(self.sci, "wb") as f:
                f.write(struct.pack(HDRFMT, MAGIC, VERSION, 0, RECSZ, 0))
            open(self.sgt, "wb").close()
        self._sgtf = open(self.sgt, "r+b")

    # ---------- skriv ----------
    def add(self, spg_text: str, verify: bool = True) -> int:
        """Legg til éit parti (SPG-tekst). Returnerer game-id.

        verify: spel partiet opp att mot reglane (game_from_spg_any) og bruk
        den verifiserte plies-teljinga — dyrest, men indeksens tal blir sanne.
        """
        tags = _tags_of(spg_text)
        plies = None
        if verify:
            from .spg import game_from_spg_any
            g = game_from_spg_any(spg_text)
            st = g["state"] if isinstance(g, dict) and "state" in g else None
            plies = int(tags.get("Ply", 0)) if "Ply" in tags else None
            if plies is None:
                # tel trekklinjene (same som eval_match tel halvtrekk)
                plies = _ply_count(spg_text)
        blob = zlib.compress(spg_text.encode("utf-8"), 9)
        offset = self._sgtf.seek(0, 2)          # end-of-file
        self._sgtf.write(blob)
        self._sgtf.flush()
        rec = struct.pack(
            RECFMT, self.count, offset, len(blob), len(spg_text.encode("utf-8")),
            RESULT_CODE.get(tags.get("Result", "*"), 0),
            1 if _is_nesseby(tags) else 0,
            min(plies or 0, 65535),
            _date_int(tags), 0, 0,
            _pack(tags.get("Event", ""), 40),
            _pack(tags.get("Menn", ""), 24),
            _pack(tags.get("Koner", ""), 24),
            _pack(tags.get("Termination", ""), 12),
            _pack(tags.get("Variant", ""), 12))
        self._records += rec
        self.count += 1
        self._flush_index()
        return self.count - 1

    def import_file(self, path: str, verify: bool = True) -> list[int]:
        """Importerer alle partia i éi SPG-fil. Returnerer game-idane."""
        games = _split_games(open(path, encoding="utf-8").read())
        return [self.add(g, verify=verify) for g in games]

    def _flush_index(self):
        with open(self.sci, "wb") as f:
            f.write(struct.pack(HDRFMT, MAGIC, VERSION, self.count, RECSZ, 0))
            f.write(self._records)

    # ---------- les ----------
    def record(self, gid: int) -> Game:
        if gid < 0 or gid >= self.count:
            raise IndexError(f"game {gid} out of range (db has {self.count})")
        return Game(self._records[gid * RECSZ:(gid + 1) * RECSZ], gid)

    def get(self, gid: int) -> str:
        rec = self.record(gid)
        self._sgtf.seek(rec.offset)
        blob = self._sgtf.read(rec.clen)
        txt = zlib.decompress(blob).decode("utf-8")
        if len(txt.encode("utf-8")) != rec.rlen:
            raise RuntimeError(f"game {gid}: raw length mismatch "
                               f"({len(txt.encode('utf-8'))} != {rec.rlen})")
        return txt

    def list(self, result: str | None = None, nesseby: bool | None = None,
             event: str | None = None, min_plies: int = 0,
             after: int = 0) -> list[Game]:
        """Filtrering berre i indekspostane — datafila vert ikkje rørt."""
        out = []
        for i in range(self.count):
            g = self.record(i)
            if result is not None and g.result_str != result:
                continue
            if nesseby is not None and g.nesseby != nesseby:
                continue
            if event and event.lower() not in g.event.lower():
                continue
            if g.plies < min_plies:
                continue
            if after and g.date <= after:
                continue
            out.append(g)
        return out

    def stats(self) -> dict:
        by_res = {1: 0, 2: 0, 3: 0, 0: 0}
        nb = 0
        plies = []
        for i in range(self.count):
            g = self.record(i)
            by_res[g.result] = by_res.get(g.result, 0) + 1
            nb += g.nesseby
            plies.append(g.plies)
        return {"games": self.count,
                "menn_wins": by_res.get(RES_MENN, 0),
                "giera_wins": by_res.get(RES_GIERA, 0),
                "draws": by_res.get(RES_DRAW, 0),
                "nesseby": nb,
                "plies_min": min(plies) if plies else 0,
                "plies_max": max(plies) if plies else 0,
                "plies_mean": (sum(plies) / len(plies)) if plies else 0,
                "sci_bytes": os.path.getsize(self.sci),
                "sgt_bytes": os.path.getsize(self.sgt)}

    def compact(self):
        """Skriv datafila på nytt med berre noverande parti (SCID: reindex)."""
        blobs = []
        for i in range(self.count):
            g = self.record(i)
            self._sgtf.seek(g.offset)
            blobs.append(self._sgtf.read(g.clen))
        self._sgtf.close()
        with open(self.sgt, "wb") as f:
            recs = b""
            off = 0
            for i, b in enumerate(blobs):
                g = self.record(i)
                f.write(b)
                rec = struct.pack(RECFMT, i, off, len(b), g.rlen, g.result,
                                  g.nesseby, g.plies, g.date, 0, 0,
                                  _pack(g.event, 40), _pack(g.white, 24),
                                  _pack(g.black, 24), _pack(g.termination, 12),
                                  _pack(g.variant, 12))
                recs += rec
                off += len(b)
        self._records = recs
        self._flush_index()
        self._sgtf = open(self.sgt, "r+b")

    def close(self):
        if self._sgtf:
            self._sgtf.close()
            self._sgtf = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _is_nesseby(tags: dict) -> bool:
    v = str(tags.get("Variant", "")) + " " + str(tags.get("nesseby", ""))
    return "nesseby" in v.lower()


def _split_games(text: str) -> list[str]:
    """Dele ei SPG-fil med fleire parti på same måte som verifiseraren:
    kvart parti startar med si eiga [Event]-tag (eller SPG-V1 etter første)."""
    # Eit parti er [tag-hovud …, SPG-V1, trekklener …]. Skiljet går først
    # når ein ny tag-header kjem etter at blokken har sett SPG-V1.
    blocks = []
    cur = []
    seen = False
    for line in text.splitlines(keepends=True):
        s = line.strip()
        if s == "SPG-V1":
            seen = True
        elif cur and seen and TAG_RE.match(s):
            blocks.append("".join(cur))
            cur, seen = [], False
        cur.append(line)
    if cur:
        blocks.append("".join(cur))
    return [b for b in blocks if "SPG-V1" in b]


def open_db(prefix: str) -> GameDB:
    return GameDB(prefix)
