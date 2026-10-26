// Parity probe: replay a position stream through the REFERENCE JS engine.
// stdin:  {"positions":[{"fen":…,"pool":["M A1 B1",…],"mv":"M A1 B1"}, …]}
// stdout: [{"fen":…,"pool":[…],"result":{"mv","fen","winner"}|{"mv","error"}}, …]
//
// Canonical move identity: "LETTER FROM TO" — letter M/K (soldier by side),
// Q queen, k king. FEN keeps soldiers anonymous within a stack, so the
// letter class is the exact identity FEN preserves; square tokens alone are
// ambiguous when queen+king share a square, and pids are not comparable
// across FEN re-assignment. The probe finds the legal move in JS whose
// canonical encoding equals pos.mv, applies it, and returns the FEN.
//
// Usage: node tools/parity_probe.js < probes/in.json > probes/out.json
"use strict";
const path = require("path");
const root = path.join(__dirname, "..", "vendor");
require(path.join(root, "engine.js"));
require(path.join(root, "fen.js"));
const SS = global.SS;

function encL(st, mv) {
  const p = SS.piece(st, mv.pid);
  const L = p.t === "q" ? "Q" : p.t === "k" ? "k" : (p.o === 0 ? "M" : "K");
  let frm, to;
  if (p.t === "s") {
    frm = SS.soldierAt(p.o, p.at);
    to = SS.soldierAt(p.o, p.at + (mv.kind === "rel" ? 1 : mv.v));
  } else if (mv.to !== undefined && mv.to !== null) {
    frm = p.at; to = mv.to;
  } else {
    const r = SS.pathFor(st, p, mv.dir, mv.v);
    if (!r.legal || !r.path.length) return null;
    frm = p.at;
    to = r.path[r.path.length - 1];
  }
  return L + " " + SS.name(frm) + " " + SS.name(to);
}

const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = [];
for (const pos of input.positions) {
  const rec = { fen: pos.fen, pool: [], result: null };
  let st;
  try {
    st = SS.fromFEN(pos.fen);
  } catch (e) {
    rec.error = String(e.message || e);
    out.push(rec);
    continue;
  }
  const seen = new Set();
  SS.legalMoves(st, st.turn).forEach((mv) => {
    const s = encL(st, mv);
    if (s && !seen.has(s)) { seen.add(s); rec.pool.push(s); }
  });
  if (pos.mv) {
    const clone = SS.deserialize(SS.serialize(st));
    // S-UCI §4: stablar løysast til den FRØMSTE brikka (høgast syklusposisjon),
    // ikkje «fyrste treff i legalMoves-rekkjefylgja».
    let mv = null, bestAt = -Infinity;
    for (const m of SS.legalMoves(clone, clone.turn)) {
      if (encL(clone, m) !== pos.mv) continue;
      const pm = SS.piece(clone, m.pid);
      if (pm.at > bestAt) { bestAt = pm.at; mv = m; }
    }
    if (!mv) {
      rec.result = { mv: pos.mv, error: "ulovleg" };
    } else {
      SS.applyMove(clone, mv);
      SS.settle(clone);
      rec.result = { mv: pos.mv, fen: SS.toFEN(clone), winner: clone.winner };
    }
  }
  out.push(rec);
}
process.stdout.write(JSON.stringify(out));
