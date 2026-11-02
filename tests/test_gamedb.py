"""Testar den indekterte speltdatabasen (SCID-metoden: .sci + .sgt).

Køyrd som:  .venv/bin/python -m tests.test_gamedb
Krev:     examples/ sin SPG (eller ein generert tekst); lagar DB i tmp.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sahkku_ai import gamedb

SPG_TWO = """\
[Event "test one"]
[Date "2026-11-02"]
[Menn "Menn — A"]
[Koner "Koner — B"]
[Result "1-0"]
[Termination "dronning"]
SPG-V1

1. open: M(X II X) K(II X III) -> M
1. M: II III X
   M->-
2. M: X
   M(A5)->C1
2. K: X II III
   K(D2)->C2
*

[Event "test two"]
[Date "2026-11-03"]
[Menn "Menn — C"]
[Koner "Koner — D"]
[Result "0-1"]
[Variant "Nesseby"]
SPG-V1

1. open: M(II X X) K(X II X) -> K
1. K: X II X
   K->-
2. K: III
   K(B1)->B2
2. M: X II III
   M(A2)->B2
*
"""


def main():
    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, "games")
        # split logic
        games = gamedb._split_games(SPG_TWO)
        assert len(games) == 2, f"split: {len(games)}"

        db = gamedb.open_db(prefix)
        assert db.count == 0
        ids = [db.add(g, verify=False) for g in games]
        assert ids == [0, 1]
        # round-trip: SPG-tekst byte-identisk
        for gid, g in zip(ids, games):
            assert db.get(gid) == g, f"roundtrip gid {gid}"
        # index fields
        g0 = db.record(0)
        assert g0.result_str == "1-0" and not g0.nesseby and g0.date == 20261102
        assert g0.event == "test one" and g0.white == "Menn — A"
        g1 = db.record(1)
        assert g1.result_str == "0-1" and g1.nesseby and g1.date == 20261103
        # filtering on the index alone
        assert [g.gid for g in db.list(result="1-0")] == [0]
        assert [g.gid for g in db.list(nesseby=True)] == [1]
        assert [g.gid for g in db.list(event="test")] == [0, 1]
        assert db.list(result="*") == []
        # reopen (persistence), append, size check
        db.close()
        db = gamedb.open_db(prefix)
        assert db.count == 2
        assert db.get(0) == games[0]
        db.add(games[1], verify=False)
        assert db.count == 3 and db.record(2).nesseby
        # out of range
        try:
            db.get(99)
            raise AssertionError("get(99) should raise")
        except IndexError:
            pass
        # compact keeps every game readable
        db.compact()
        assert db.get(2) == games[1]
        s = db.stats()
        assert s["games"] == 3 and s["nesseby"] == 2
        db.close()
        # truncated data file is detected
        with open(prefix + ".sgt", "r+b") as f:
            f.truncate(2)
        try:
            gamedb.open_db(prefix)
            raise AssertionError("truncated .sgt should raise")
        except SystemExit:
            pass

    print("gamedb: all checks OK (split, roundtrip, index fields, filters, "
          "persist, append, compact, truncation detection)")


if __name__ == "__main__":
    main()
