"""CLI for the indexed game database (SCID method: .sci index + .sgt data).

  import  <db-prefix> <games.spg>     append every game in the file
  stats   <db-prefix>                 counts and sizes from the index alone
  list    <db-prefix> [--result 1-0] [--nesseby] [--event STR] [--min-plies N]
  get     <db-prefix> <gid>           print the SPG text of one game
  export  <db-prefix> <out.spg> [--result ...] [--nesseby] [--event STR]
  compact <db-prefix>                 rewrite the data file (reindex)

All filtering runs on the fixed-width index records; the data file is only
touched for `get`/`export`. Output is English (project rule for Python CLIs).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sahkku_ai import gamedb


def main():
    ap = argparse.ArgumentParser(prog="game_db",
                                 description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("import", help="append every game from an SPG file")
    p.add_argument("prefix")
    p.add_argument("spg")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the rules replay (faster, counts from the text)")

    for name in ("stats", "compact"):
        p = sub.add_parser(name)
        p.add_argument("prefix")

    p = sub.add_parser("list", help="filter games by index fields")
    p.add_argument("prefix")
    p.add_argument("--result", choices=["1-0", "0-1", "1/2-1/2", "*"])
    p.add_argument("--nesseby", action="store_true")
    p.add_argument("--standard", action="store_true", help="only non-Nesseby games")
    p.add_argument("--event")
    p.add_argument("--min-plies", type=int, default=0)

    p = sub.add_parser("get", help="print one game's SPG text")
    p.add_argument("prefix")
    p.add_argument("gid", type=int)

    p = sub.add_parser("export", help="write matching games back out as one SPG file")
    p.add_argument("prefix")
    p.add_argument("out")
    p.add_argument("--result", choices=["1-0", "0-1", "1/2-1/2", "*"])
    p.add_argument("--nesseby", action="store_true")
    p.add_argument("--standard", action="store_true")
    p.add_argument("--event")

    a = ap.parse_args()
    db = gamedb.open_db(a.prefix)
    try:
        if a.cmd == "import":
            ids = db.import_file(a.spg, verify=not a.no_verify)
            print(f"imported {len(ids)} games ({ids[0]}..{ids[-1]}) into {a.prefix}.sci")
        elif a.cmd == "stats":
            s = db.stats()
            for k, v in s.items():
                print(f"{k}: {v}")
        elif a.cmd == "list":
            nb = True if a.nesseby else (False if a.standard else None)
            rows = db.list(result=a.result, nesseby=nb, event=a.event,
                           min_plies=a.min_plies)
            for g in rows:
                nbtxt = "Nesseby" if g.nesseby else "standard"
                print(f"{g.gid:5d}  {g.result_str:8s} {g.plies:4d}p  {nbtxt:8s} "
                      f"{g.date:8d}  {g.white} - {g.black}  [{g.event}]")
            print(f"({len(rows)} of {db.count} games)")
        elif a.cmd == "get":
            sys.stdout.write(db.get(a.gid))
        elif a.cmd == "export":
            nb = True if a.nesseby else (False if a.standard else None)
            rows = db.list(result=a.result, nesseby=nb, event=a.event)
            with open(a.out, "w", encoding="utf-8") as f:
                for g in rows:
                    f.write(db.get(g.gid))
            print(f"exported {len(rows)} games to {a.out}")
        elif a.cmd == "compact":
            db.compact()
            print(f"compacted: {db.stats()}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
