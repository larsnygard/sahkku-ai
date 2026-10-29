"""HTTP(S) front end for an S-UCI sáhkku engine, with API keys.

Why: the browser app can only run engines it can download as JavaScript. A
server lets anyone point the app at a stronger machine (or a model too large to
ship) without giving away the weights. Because that server is on the open
internet, it needs to say who may use it — hence API keys.

    python -m sahkku_ai.server --engine engines/nais2-nn.js --port 8734 \\
        --key "$(openssl rand -hex 16)" --allow-origin https://snart.com

Protocol (deliberately tiny; the app is the referee, not this server):

    GET  /health          200 {"ok":true}                  no key needed
    GET  /v1/info         200 {"name","protocol","engine"}  key needed
    POST /v1/move         200 {"move":"O1-O2","ms":412}     key needed
         body {"fen": "<S-UCI FEN>"}

Auth: `Authorization: Bearer <key>` or `X-API-Key: <key>`. Keys are compared in
constant time and never logged; a key may carry a label (`key # label`) and the
label is what appears in the log.

The move token is what S-UCI already returns (`bestmove`), e.g. `O1-O2` or
`X->O1`; the client matches it against its own legal pool, so an engine that
hallucinates a move cannot corrupt a game.

TLS: this speaks plain HTTP. Put it behind a reverse proxy (Caddy/nginx) for
https, which is also where you would terminate certificates.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL = 1
MAX_BODY = 4096
DEFAULT_RATE = 120          # requests per minute per key


# ---------------------------------------------------------------- engine glue
class EngineError(RuntimeError):
    pass


class SuciEngine:
    """One long-lived S-UCI subprocess, used strictly one request at a time."""

    def __init__(self, engine: str, timeout: float = 30.0):
        self.timeout = timeout
        self.lock = threading.Lock()
        cmd = ["node", engine] if engine.endswith(".js") else engine.split()
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.name = "unknown"
        self._handshake()

    def _readline(self) -> str:
        line = self.proc.stdout.readline()
        if not line:
            raise EngineError("engine closed the pipe")
        return line.strip()

    def _send(self, cmd: str) -> None:
        if self.proc.stdin is None:
            raise EngineError("no stdin")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _handshake(self) -> None:
        deadline = time.time() + self.timeout
        self._send("uci")
        while time.time() < deadline:
            line = self._readline()
            if line.startswith("id name"):
                self.name = line[len("id name"):].strip()
            if line == "uciok":
                self._send("isready")
                while time.time() < deadline:
                    if self._readline() == "readyok":
                        self._send("ucinewgame")
                        return
                break
        raise EngineError("engine did not complete the UCI handshake")

    def bestmove(self, fen: str) -> tuple[str, int]:
        with self.lock:
            t0 = time.time()
            self._send("position fen " + fen)
            self._send("go")
            deadline = t0 + self.timeout
            while time.time() < deadline:
                line = self._readline()
                if line.startswith("bestmove"):
                    parts = line.split()
                    tok = parts[1] if len(parts) > 1 else "none"
                    return tok, int((time.time() - t0) * 1000)
                if line.startswith("error"):
                    raise EngineError(line)
            raise EngineError("engine timed out")

    def close(self) -> None:
        try:
            self._send("quit")
        except Exception:                      # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=3)
        except Exception:                      # noqa: BLE001
            self.proc.kill()


# ------------------------------------------------------------------- key store
class Keys:
    """Keys are held in memory with a label; comparisons are constant time."""

    def __init__(self, entries: list[tuple[str, str]]):
        self.entries = [(k, label) for k, label in entries if k]
        self.rate = DEFAULT_RATE
        self.hits: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}

    def __bool__(self) -> bool:
        return bool(self.entries)

    def label_for(self, presented: str | None) -> str | None:
        """Return the key's label, or None. Constant time over every stored key."""
        if not presented:
            return None
        found = None
        for key, label in self.entries:
            if hmac.compare_digest(key, presented):
                found = label
        return found

    def allow(self, label: str) -> bool:
        """Simple per-key sliding window: n requests per 60 s."""
        if self.rate <= 0:
            return True
        now = time.time()
        win = [t for t in self.hits.get(label, []) if now - t < 60]
        if len(win) >= self.rate:
            self.hits[label] = win
            return False
        win.append(now)
        self.hits[label] = win
        return True

    def count(self, label: str) -> int:
        self.counts[label] = self.counts.get(label, 0) + 1
        return self.counts[label]


def load_keys(args) -> Keys:
    entries: list[tuple[str, str]] = []
    for k in args.key or []:
        entries.append((k, "key-" + str(len(entries) + 1)))
    for path in args.key_file or []:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, label = line.partition("#")
                key = key.strip()
                label = label.strip() or ("key-" + str(len(entries) + 1))
                entries.append((key, label))
    env = os.environ.get("SAHKKU_API_KEYS", "")
    for i, key in enumerate([x.strip() for x in env.split(",") if x.strip()]):
        entries.append((key, f"env-{i + 1}"))
    keys = Keys(entries)
    keys.rate = args.rate
    return keys


# ---------------------------------------------------------------------- server
def make_handler(engine: SuciEngine, keys: Keys, allow_origin: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"SnartSahkku-engine/{PROTOCOL}"

        # -- helpers ------------------------------------------------------
        def log_message(self, fmt, *a):                     # noqa: A003
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", allow_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers",
                             "Authorization, Content-Type, X-API-Key")
            self.send_header("Access-Control-Max-Age", "600")
            if allow_origin != "*":
                self.send_header("Vary", "Origin")

        def _json(self, code: int, payload: dict, extra: dict | None = None) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _presented_key(self) -> str | None:
            auth = self.headers.get("Authorization", "")
            if auth.lower().startswith("bearer "):
                return auth[7:].strip()
            return self.headers.get("X-API-Key")

        def _check_key(self) -> str | None:
            """Label when authorised; sends the 401 itself and returns None otherwise."""
            label = keys.label_for(self._presented_key())
            if label is None:
                self._json(401, {"error": "unauthorized",
                                 "hint": "send Authorization: Bearer <key> or X-API-Key"},
                           {"WWW-Authenticate": 'Bearer realm="sahkku"'})
                return None
            if not keys.allow(label):
                self._json(429, {"error": "rate_limited", "limit_per_min": keys.rate},
                           {"Retry-After": "60"})
                return None
            return label

        # -- verbs --------------------------------------------------------
        def do_OPTIONS(self):                               # noqa: N802
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):                                   # noqa: N802
            if self.path.rstrip("/") == "/health":
                return self._json(200, {"ok": True})
            if self.path.rstrip("/") == "/v1/info":
                label = self._check_key()
                if label is None:
                    return
                keys.count(label)
                return self._json(200, {"name": engine.name, "protocol": PROTOCOL,
                                        "engine": "suci", "variant": "sahkku+nesseby"})
            return self._json(404, {"error": "not_found"})

        def do_POST(self):                                  # noqa: N802
            if self.path.rstrip("/") != "/v1/move":
                return self._json(404, {"error": "not_found"})
            label = self._check_key()
            if label is None:
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return self._json(400, {"error": "bad_length", "max": MAX_BODY})
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad_json"})
            fen = (data.get("fen") or "").strip()
            if not fen or len(fen.split()) < 4:
                return self._json(400, {"error": "bad_fen",
                                        "hint": "send the S-UCI FEN as {\"fen\": \"...\"}"})
            try:
                move, ms = engine.bestmove(fen)
            except EngineError as e:
                return self._json(502, {"error": "engine_failed", "detail": str(e)[:200]})
            n = keys.count(label)
            self.log_message("key=%s req=%d move=%s %dms", label, n, move, ms)
            return self._json(200, {"move": move, "ms": ms, "engine": engine.name})

    return Handler


def serve(args) -> None:
    keys = load_keys(args)
    if not keys:
        print("warning: no API keys configured — the server is OPEN to anyone.\n"
              "         Set --key/--key-file/SAHKKU_API_KEYS to restrict access.",
              file=sys.stderr)
    engine = SuciEngine(args.engine)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(engine, keys, args.allow_origin))
    httpd.daemon_threads = True
    print(f"engine: {engine.name}  (protocol {PROTOCOL})")
    print(f"listening on http://{args.host}:{args.port}  origin={args.allow_origin}  "
          f"keys={len(keys.entries) or 'none'}  rate={keys.rate}/min")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
        httpd.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="engines/nais2-nn.js",
                    help="S-UCI engine: a .js file (run with node) or a command line")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose it")
    ap.add_argument("--port", type=int, default=8734)
    ap.add_argument("--key", action="append", help="an API key (repeatable)")
    ap.add_argument("--key-file", action="append",
                    help="file with one key per line, optionally «key # label»")
    ap.add_argument("--rate", type=int, default=DEFAULT_RATE,
                    help=f"requests per minute per key (default {DEFAULT_RATE}, 0 = no limit)")
    ap.add_argument("--allow-origin", default="*",
                    help="CORS origin for the browser (default *, or e.g. https://snart.com)")
    serve(ap.parse_args())


if __name__ == "__main__":
    main()
