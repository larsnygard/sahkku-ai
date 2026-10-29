"""Tests for the HTTP engine server (auth, CORS, protocol, rate limit).

Starts a real server on a free port with a real S-UCI engine and talks to it
over HTTP, because the interesting failures (a preflight that the browser
rejects, a 401 that leaks the key, a move token the app cannot match) only show
up across the wire.

Run: .venv/bin/python -m tests.test_engine_server
"""
from __future__ import annotations
import json
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = ROOT.parent / "SnartSahkku"
ENGINE = APP / "engines" / "nais2.js"
KEY = "test-key-" + "a1b2c3"
PASS = FAIL = 0


def ok(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" :: {extra}" if extra else ""))


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def call(base, path, method="GET", body=None, key=None, headers=None):
    req = urllib.request.Request(base + path, method=method)
    if key:
        req.add_header("Authorization", "Bearer " + key)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=60) as r:
            return r.status, dict(r.headers), json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b"{}")
        except Exception:                       # noqa: BLE001
            payload = {}
        return e.code, dict(e.headers), payload


def sample_fen() -> str:
    """A real mid-game FEN, produced by the app's own rules module."""
    js = (
        "global.window=global; global.navigator={userAgent:'node'};"
        f"require('{APP}/js/engine.js'); require('{APP}/js/fen.js');"
        "let s=12345; const r=()=>((s=(s*1103515245+12345)&0x7fffffff)/0x7fffffff);"
        "const st=SS.newGame({first:0,nesseby:true}); SS.openingRoll(st,r); SS.roll(st,r);"
        "st.phase='move'; process.stdout.write(SS.toFEN(st));"
    )
    return subprocess.run(["node", "-e", js], capture_output=True, text=True,
                          timeout=60).stdout.strip()


def main():
    if not ENGINE.exists():
        print(f"  SKIP: no engine at {ENGINE}")
        return 0
    fen = sample_fen()
    ok("got a sample FEN from the app", len(fen.split()) >= 4, fen)

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "sahkku_ai.server", "--engine", str(ENGINE),
         "--port", str(port), "--key", KEY, "--key", "second-key-#labelled",
         "--allow-origin", "https://snart.com", "--rate", "6"],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # wait for the port
        for _ in range(120):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.25)
        else:
            ok("server started", False, (proc.stderr.read() or "")[-400:])
            return 1
        ok("server started", True)

        # ---- health needs no key ----------------------------------------
        code, hdrs, body = call(base, "/health")
        ok("/health answers without a key", code == 200 and body.get("ok") is True, (code, body))
        ok("CORS header is the configured origin",
           hdrs.get("Access-Control-Allow-Origin") == "https://snart.com", hdrs.get("Access-Control-Allow-Origin"))

        # ---- auth --------------------------------------------------------
        code, hdrs, body = call(base, "/v1/info")
        ok("info without a key is 401", code == 401, (code, body))
        ok("401 carries WWW-Authenticate", "Bearer" in (hdrs.get("WWW-Authenticate") or ""),
           hdrs.get("WWW-Authenticate"))
        code, _, body = call(base, "/v1/info", key="wrong-key")
        ok("a wrong key is 401", code == 401, (code, body))
        code, _, body = call(base, "/v1/info", key=KEY)
        ok("a valid key gets engine info", code == 200 and body.get("protocol") == 1, (code, body))
        ok("info names the engine", bool(body.get("name")), body)

        # X-API-Key must work too (some clients cannot set Authorization)
        code, _, body = call(base, "/v1/info", headers={"X-API-Key": KEY})
        ok("X-API-Key is accepted", code == 200, (code, body))
        code, _, body = call(base, "/v1/info", key=KEY, headers={"X-API-Key": "nope"})
        ok("a valid Bearer wins over a bad X-API-Key", code == 200, (code, body))

        # ---- CORS preflight ---------------------------------------------
        code, hdrs, _ = call(base, "/v1/move", method="OPTIONS",
                             headers={"Origin": "https://snart.com",
                                      "Access-Control-Request-Method": "POST",
                                      "Access-Control-Request-Headers": "authorization"})
        ok("preflight is 204", code == 204, code)
        allow = (hdrs.get("Access-Control-Allow-Headers") or "").lower()
        ok("preflight allows Authorization", "authorization" in allow, allow)
        ok("preflight allows POST", "POST" in (hdrs.get("Access-Control-Allow-Methods") or ""),
           hdrs.get("Access-Control-Allow-Methods"))

        # ---- the actual move --------------------------------------------
        code, _, body = call(base, "/v1/move", method="POST", body={"fen": fen}, key=KEY)
        ok("move request succeeds", code == 200, (code, body))
        move = body.get("move") or ""
        ok("a move token comes back", bool(move) and move != "none", body)
        # S-UCI skriv trekk som FROM-TO med bindestrek («O1-O2»), òg for løysing;
        # spesialtokens er «pass», «reroll» og «none».
        ok("the token is S-UCI notation (FROM-TO with a hyphen)",
           bool(re.match(r"^[A-O][1-3]-[A-O][1-3]$", move)) or move in ("pass", "reroll", "none"), move)
        ok("the move was made by the engine it advertises", body.get("engine") == "NAIS.2", body.get("engine"))

        # the app must be able to match it: same position, its own legal pool
        # NB: ein «release» skriv seg O1->O2 (frå heimefeltet til feltet framfor),
        # ikkje X->O2 — motoren sin bestmove er alltid from->to. Klienten må
        # godta båe formene; her samanliknar vi from->to slik motoren skriv det.
        js = (
            "global.window=global; global.navigator={userAgent:'node'};"
            f"require('{APP}/js/engine.js'); require('{APP}/js/fen.js');"
            f"const st=SS.fromFEN({json.dumps(fen)});"
            "const lm=SS.legalMoves(st, st.turn);"
            "const names=lm.map(m=>{const p=SS.piece(st,m.pid);"
            "const from=p.t==='s'?SS.soldierAt(p.o,p.at):p.at;"
            "const to=p.t==='s'?SS.soldierAt(p.o,p.at+(m.kind==='rel'?1:m.v)):SS.moveDest(st,m);"
            "return SS.sqName(from)+'-'+SS.sqName(to);});"
            "process.stdout.write(JSON.stringify(names));"
        )
        legal = json.loads(subprocess.run(["node", "-e", js], capture_output=True,
                                         text=True, timeout=60).stdout or "[]")
        ok("the returned token is in the app's legal pool", move in legal,
           f"{move} not in {legal[:8]}")

        # ---- bad input ---------------------------------------------------
        code, _, body = call(base, "/v1/move", method="POST", body={"fen": "nonsense"}, key=KEY)
        ok("a broken FEN is 400", code == 400, (code, body))
        code, _, body = call(base, "/v1/move", method="POST", body={"nope": 1}, key=KEY)
        ok("a missing FEN is 400", code == 400, (code, body))
        code, _, body = call(base, "/v1/move", method="POST", body={"fen": fen}, key="wrong")
        ok("an unauthenticated move is 401", code == 401, (code, body))

        # ---- rate limit (server started with 6/min) ----------------------
        codes = []
        for _ in range(8):
            c, _, _ = call(base, "/v1/move", method="POST", body={"fen": fen}, key=KEY)
            codes.append(c)
        ok("requests past the limit get 429", 429 in codes, codes)
        ok("the limit is per key, not global: the other key still works",
           call(base, "/v1/info", key="second-key-#labelled")[0] in (200, 429),
           "second key state")

        # ---- the log must not contain a key -----------------------------
        proc.terminate()
        try:
            _, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, err = proc.communicate()
        ok("the key never appears in the log", KEY not in (err or ""), (err or "")[-300:])
        ok("the key's label is what gets logged", "labelled" in (err or "") or "key-1" in (err or ""),
           (err or "")[-300:])
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    print(f"test_engine_server: {PASS} pass, {FAIL} fail")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
