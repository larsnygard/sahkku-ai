/* sahkku-nn — S-UCI 1.0 neural motor for sáhkku (spec: docs/SAHKKU-UCI.md).
 * Reglane er DEN EKAKTE JS-motoren: js/engine.js + js/fen.js frå SnartSahkku
 * blir INJISERTE under ved `python -m sahkku_ai.export` (§RULES§/§FEN§-markørar)
 * — ingen avskriven kopi, slik at reglane ALLTID er 1:1 med DOM-en.
 * Nettopp: __WEIGHTS_FILE__ (JSON, base64 float32), køyrd i rein JS her —
 * same lag som treningsramverket si PyTorch-modell (conv-trunk + res-blokker,
 * policy 8282 + value tanh). Ingen npm-avhengnader: Node 18+.
 */
'use strict';
var NN = (function () {
var M = {};
'use strict';

// ==================== RULES + FEN (injisert ved eksport) ==================
// engine.js/fen.js er IIFAr som legg seg på globalThis.SS — det er berre å
// plukke dei opp att etter injeksjonen.
/*§RULES§*/
/*§FEN§*/
var SS = (typeof globalThis !== 'undefined' ? globalThis : this).SS;

// ==================== FEATURES + ACTIONS (mirrors features.py) ============
var N_PLANES = 16, N_GLOBAL = __N_GLOBAL__;   // fylt inn av export.py
var REL_BASE = 0, MOVE_BASE = 180, SPEC_BASE = 180 + 8100;
var CLASSES = ['M', 'K', 'Q', 'k'];
var FIRST_PASS = [[], []];
function firstOf(pl, sq) {
  var m = FIRST_PASS[pl];
  if (!m.length) {
    for (var t = 0; t < 60; t++) {
      var s = SS.soldierAt(pl, t);
      if (m[s] === undefined) m[s] = t;
    }
  }
  return m[sq];
}
function encodeState(st) {
  var pl = new Float32Array(N_PLANES * 45);
  function S(i, r, c) { pl[i * 45 + r * 15 + c] = 1; }
  function addc(i, r, c, v) { pl[i * 45 + r * 15 + c] += v; }
  for (var i = 0; i < st.pieces.length; i++) {
    var p = st.pieces[i];
    if (p.captured) continue;
    var sq = p.t === 's' ? SS.soldierAt(p.o, p.at) : p.at;
    var q = SS.pt(sq), r = q.row, c = q.col;
    if (p.t === 's') {
      if (!p.released) S(2 + p.o, r, c);
      else {
        S(p.o, r, c);
        addc(12 + p.o, r, c, 1 / 15);
        if (r === 1) {
          var f = firstOf(p.o, sq);
          if (f !== undefined && p.at % 60 !== f) S(14 + p.o, r, c);
        }
      }
    } else if (p.t === 'q') {
      S(p.released ? (4 + p.o) : (6 + p.o), r, c);
    } else if (p.t === 'k') {
      S(p.o === 0 ? 8 : (p.o === 1 ? 9 : 10), r, c);
    }
  }
  var rk = SS.activeRank(st), v = SS.activeValue(st);
  if (st.phase === 'move' && rk !== null && rk !== undefined) {
    if (rk === SS.RANK.X) S(11, 0, 0);
    else if (rk === SS.RANK.III) S(11, 0, 1);
    else S(11, 0, 2);
  }
  var g = new Float32Array(N_GLOBAL);
  var me = st.turn, foe = 1 - me;
  g[0] = st.turn === 0 ? 1 : -1;
  g[1] = st.kills[me] / 15;                       // POV-relativt (mi side fyrst)
  g[2] = st.kills[foe] / 15;
  // UTLØYSTE soldatar, ikkje «i live»: heimebrikker er trygge og gjev ingen
  // mobilitet. Talet på heimebrikker er implisitt 15 − utløyste − slagtekne.
  g[3] = SS.aliveSoldiers(st, me).filter(function (p) { return p.released; }).length / 15;
  g[4] = SS.aliveSoldiers(st, foe).filter(function (p) { return p.released; }).length / 15;
  g[5] = Math.min(st.moves, 200) / 200;
  var qm = SS.queen(st, me), qf = SS.queen(st, foe);
  g[6] = (qm && qm.released) ? 1 : 0;   // UTLØYST og POV-relativ (jf. features.py)
  g[7] = (qf && qf.released) ? 1 : 0;
  g[8] = (st.dice ? st.dice.filter(function (d) { return d === 'X'; }).length : 0) / 3;
  g[9] = st.winner === null || st.winner === undefined ? 1 : (st.winner === st.turn ? 1 : -1);

  // ---- kongedoktrinen + terningane (spegl av features.py) --------------
  // NB: alt nedanfor må reknast UTAN å endre stillinga til kallaren.
  var k = SS.king(st);
  var mine = (k && k.o === me), theirs = (k && k.o === foe);
  g[10] = mine ? 1 : 0;
  g[11] = theirs ? 1 : 0;
  if (mine) {
    var foes = SS.aliveSoldiers(st, foe);
    var best = 99;
    for (var fi = 0; fi < foes.length; fi++) {
      var fs = SS.soldierAt(foes[fi].o, foes[fi].at);
      var d = cheb(k.at, fs);
      if (d < best) best = d;
    }
    if (best < 99) g[12] = 1 - Math.min(best, 14) / 14;
  }
  var us = usableAndCapture(st, foe);
  if (qm && qm.released && !qm.captured) {
    g[13] = us.caps[qm.at] ? 1 : 0;              // dronninga er takeleg NO
  }
  g[14] = DICE_USE_PROB[us.mask];                 // friheit i neste kast
  g[15] = st.nesseby ? 1 : 0;                     // regelmodus (Nesseby av/på)
  return { planes: pl, glob: g };
}

function cheb(a, b) {
  var pa = SS.pt(a), pb = SS.pt(b);
  return Math.max(Math.abs(pa.col - pb.col), Math.abs(pa.row - pb.row));
}

// Sannsynet for at eit kast har minst éin terning sida kan nytte, gjeve
// kva fjes ho kan nytte i det heile (bit 0 = X, bit 1 = II, bit 2 = III).
var DICE_USE_PROB = [0.0, 0.5781, 0.5781, 0.8750, 0.5781, 0.8750, 0.8750, 0.9844];

// Kva terningverdiar kan `side` nytte, og kva felt kan ho slå på no?
// (same som _usable_mask/_capture_squares i features.py)
function usableAndCapture(st, side) {
  var save = { dice: st.dice ? st.dice.slice() : null,
               used: st.used ? st.used.slice() : null,
               phase: st.phase, turn: st.turn };
  var mask = 0, caps = {};
  try {
    st.turn = side; st.phase = 'move';
    var faces = [['X', 1], ['II', 2], ['III', 3]];
    for (var i = 0; i < 3; i++) {
      st.dice = [faces[i][0], '_', '_'];
      st.used = [false, true, true];
      var pool = SS.legalMoves(st, side);
      for (var j = 0; j < pool.length; j++) {
        var mv = pool[j];
        if (mv.v === faces[i][1]) mask |= (1 << (faces[i][1] - 1));
        var land = SS.moveDest(st, mv);
        var on = SS.occ(st, land);
        for (var oi = 0; oi < on.length; oi++) {
          var q = on[oi];
          if (q.o != null && q.o !== side && q.id !== mv.pid) {
            caps[land] = (caps[land] || []).concat([faces[i][1]]);
          }
        }
      }
    }
  } finally {
    st.dice = save.dice; st.used = save.used; st.phase = save.phase; st.turn = save.turn;
  }
  return { mask: mask, caps: caps };
}
function letterOf(p) {
  return p.t === 'q' ? 'Q' : (p.t === 'k' ? 'k' : (p.o === 0 ? 'M' : 'K'));
}
function moveId(st, mv) {
  var p = SS.piece(st, mv.pid);
  if (mv.kind === 'rel') {
    var cls = p.t === 's' ? 0 : 1;
    var target;
    if (p.t === 's') target = SS.soldierAt(p.o, p.at + 1);
    else {
      var r1 = SS.pathFor(st, p, mv.dir, 1);
      target = r1.path[r1.path.length - 1];
    }
    return (p.o === 0 ? 0 : 90) + cls * 45 + target;
  }
  var L = letterOf(p), frm, to;
  if (p.t === 's') {
    frm = SS.soldierAt(p.o, p.at);
    to = SS.soldierAt(p.o, p.at + mv.v);
  } else if (mv.to !== undefined && mv.to !== null) {
    frm = p.at; to = mv.to;
  } else {
    var r2 = SS.pathFor(st, p, mv.dir, mv.v);
    frm = p.at; to = r2.path[r2.path.length - 1];
  }
  return MOVE_BASE + CLASSES.indexOf(L) * 2025 + frm * 45 + to;
}
function legalIds(st) {
  if (st.winner !== null) return [];
  var out = [], seen = {};
  if (st.phase === 'roll') { out.push([SPEC_BASE, { kind: 'pass' }]); return out; }
  var lms = SS.legalMoves(st, st.turn);
  for (var i = 0; i < lms.length; i++) {
    var aid = moveId(st, lms[i]);
    if (!seen[aid]) { seen[aid] = 1; out.push([aid, lms[i]]); }
  }
  if (SS.canRerollX(st)) out.push([SPEC_BASE + 1, { kind: 'reroll' }]);
  if (!out.length) out.push([SPEC_BASE, { kind: 'pass' }]);
  return out;
}

// ==================== NET (pure-JS mirror of model.py forward) ============
function b64F32(b64) {
  var bin = atob(b64);
  var a = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i);
  return new Float32Array(a.buffer);
}
var EPS = 1e-5, CFG = null, T = null;
function loadWeights(w) {
  CFG = w.meta.cfg;
  T = {};
  Object.keys(w.tensors).forEach(function (k) { T[k] = b64F32(w.tensors[k]); });
}
function conv2d(x, W, b, inC, outC, H, Wd, kh, kw, pad, out) {
  var oH = H + 2 * pad - kh + 1, oW = Wd + 2 * pad - kw + 1;
  for (var oc = 0; oc < outC; oc++) {
    var wb = oc * inC * kh * kw;
    for (var oh = 0; oh < oH; oh++) for (var ow = 0; ow < oW; ow++) {
      var s = b ? b[oc] : 0;
      for (var ic = 0; ic < inC; ic++) for (var a = 0; a < kh; a++) {
        var ih = oh + a - pad;
        if (ih < 0 || ih >= H) continue;
        for (var bb = 0; bb < kw; bb++) {
          var iw = ow + bb - pad;
          if (iw < 0 || iw >= Wd) continue;
          s += W[wb + ic * kh * kw + a * kw + bb] * x[(ic * H + ih) * Wd + iw];
        }
      }
      out[(oc * oH + oh) * oW + ow] = s;
    }
  }
  return out;
}
function bn2d(t, gamma, beta, mean, varc, ch, hw) {
  for (var c = 0; c < ch; c++) {
    var sc = gamma[c] / Math.sqrt(varc[c] + EPS), sh = beta[c] - mean[c] * sc;
    for (var i = 0; i < hw; i++) t[c * hw + i] = t[c * hw + i] * sc + sh;
  }
}
function reluInplace(t) { for (var i = 0; i < t.length; i++) if (t[i] < 0) t[i] = 0; return t; }
function linear(t, W, b, nIn, nOut) {
  var o = new Float32Array(nOut);
  for (var o2 = 0; o2 < nOut; o2++) {
    var s = b[o2], w = o2 * nIn;
    for (var i = 0; i < nIn; i++) s += W[w + i] * t[i];
    o[o2] = s;
  }
  return o;
}
function bn(pfx, ch) {
  return [T[pfx + '.weight'], T[pfx + '.bias'], T[pfx + '.running_mean'], T[pfx + '.running_var']];
}
function netForward(planes, glob) {
  var ch = CFG.ch, hw = 45;
  var x = new Float32Array(ch * hw);
  conv2d(planes, T['inp.weight'], null, N_PLANES, ch, 3, 15, 3, 3, 1, x);
  var b0 = bn('inp_bn', ch); bn2d(x, b0[0], b0[1], b0[2], b0[3], ch, hw);
  reluInplace(x);
  for (var i = 0; i < CFG.blocks; i++) {
    var pfx = 'trunk.' + i, hw2 = ch * hw;
    var y = new Float32Array(hw2), y2 = new Float32Array(hw2);
    conv2d(x, T[pfx + '.c1.weight'], null, ch, ch, 3, 15, 3, 3, 1, y);
    var b1 = bn(pfx + '.b1', ch); bn2d(y, b1[0], b1[1], b1[2], b1[3], ch, hw);
    reluInplace(y);
    conv2d(y, T[pfx + '.c2.weight'], null, ch, ch, 3, 15, 3, 3, 1, y2);
    var b2 = bn(pfx + '.b2', ch); bn2d(y2, b2[0], b2[1], b2[2], b2[3], ch, hw);
    for (var k = 0; k < hw2; k++) x[k] = y2[k] + x[k];
    reluInplace(x);
  }
  var ph = 8 * hw, pt2 = new Float32Array(ph);
  conv2d(x, T['pol_conv.weight'], T['pol_conv.bias'], ch, 8, 3, 15, 1, 1, 0, pt2);
  var pb = bn('pol_bn', 8); bn2d(pt2, pb[0], pb[1], pb[2], pb[3], 8, hw);
  reluInplace(pt2);
  var pcat = new Float32Array(ph + N_GLOBAL);
  pcat.set(pt2); pcat.set(glob, ph);
  var p1 = reluInplace(linear(pcat, T['pol_fc.weight'], T['pol_fc.bias'], ph + N_GLOBAL, CFG.fc));
  var policy = linear(p1, T['pol_out.weight'], T['pol_out.bias'], CFG.fc, SPEC_BASE + 2);
  var vh = 4 * hw, vt = new Float32Array(vh);
  conv2d(x, T['val_conv.weight'], T['val_conv.bias'], ch, 4, 3, 15, 1, 1, 0, vt);
  var vb = bn('val_bn', 4); bn2d(vt, vb[0], vb[1], vb[2], vb[3], 4, hw);
  reluInplace(vt);
  var vcat = new Float32Array(vh + N_GLOBAL);
  vcat.set(vt); vcat.set(glob, vh);
  var v1 = reluInplace(linear(vcat, T['val_fc1.weight'], T['val_fc1.bias'], vh + N_GLOBAL, CFG.fc));
  var vf = linear(v1, T['val_fc2.weight'], T['val_fc2.bias'], CFG.fc, 1);
  return { policy: policy, value: Math.tanh(vf[0]) };
}

// ==================== S-UCI LOOP (referee-checked, spec §4-5) =============
var START_FEN = 'g-g-g-g-g-g-g-g-g-g-g-g-g-g-g/.-.-.-d-.-.-.-N-.-.-.-q-.-.-./m-m-m-m-m-m-m-m-m-m-m-m-m-m-m w roll - 000 H2 0:0 0 *';
var ENGINE_NAME = '__ENGINE_NAME__';   // settast ved eksport (--name)
var st = SS.fromFEN(START_FEN);
var dbg = false, ilStreak = 0;
var LA = 4;              // 0 = rein policy (moment); k = 1-ply verdilookahead på k+1 trekk
var DEEP = 0;            // 0 = av; m > 0 = 2-ply: motstandaren sine m beste svar vert rekna med
var CAP_MARGIN = 0.12;   // trygt slag innanfor denne verdimarginen av beste trekk vert valt
var NESSBY = false;      // regelmodus når stillinga kjem frå FEN (S-UCI/eval)
var DEC = { X: 'X', '2': 'II', '3': 'III', '_': '_' };  // wire-faces -> interne fjes (som i ref)

function encode(st_, mv) {
  var p = SS.piece(st_, mv.pid);
  if (p.t === 's') {
    var from = SS.soldierAt(p.o, p.at);
    var to = SS.soldierAt(p.o, p.at + (mv.kind === 'rel' ? 1 : mv.v));
    return SS.name(from) + '-' + SS.name(to);
  }
  if (mv.to !== undefined && mv.to !== null) return SS.name(p.at) + '-' + SS.name(mv.to);
  var r = SS.pathFor(st_, p, mv.dir, mv.v);
  if (!r.legal || !r.path.length) return null;
  return SS.name(p.at) + '-' + SS.name(r.path[r.path.length - 1]);
}
function poolStrings(st_) {
  var seen = {}, o = [];
  SS.legalMoves(st_, st_.turn).forEach(function (mv) {
    var s = encode(st_, mv);
    if (s && !seen[s]) { seen[s] = 1; o.push(s); }
  });
  return o;
}
function decode(st_, s) {
  var parts = s.split('-');
  if (parts.length !== 2) return null;
  var from = SS.sqOfName(parts[0]), to = SS.sqOfName(parts[1]);
  if (from < 0 || to < 0) return null;
  var lms = SS.legalMoves(st_, st_.turn);
  for (var i = 0; i < lms.length; i++) if (encode(st_, lms[i]) === s) return lms[i];
  return null;
}
function applyToken(st_, tok, words, i) {
  if (tok === 'roll') {
    var faces = [words[i + 1], words[i + 2], words[i + 3]].map(function (f) {
      if (!DEC[f]) throw new Error('ulovleg terning «' + f + '»');
      return DEC[f];
    });
    st_.dice = faces;
    st_.used = faces.map(function (d) { return d === '_'; });
    st_.diceBy = st_.turn;
    st_.phase = 'move';
    return 4;
  }
  if (tok === 'pass') { SS.passRemaining(st_); SS.settle(st_); return 1; }
  if (tok === 'reroll') return 1;
  if (tok === 'resign') { st_.winner = 1 - st_.turn; st_.winReason = 'overgivelse'; st_.phase = 'over'; return 1; }
  if (/^[A-O][123]-[A-O][123]$/.test(tok)) {
    var mv = decode(st_, tok);
    if (!mv) throw new Error('ulovleg trekk «' + tok + '»');
    SS.applyMove(st_, mv); SS.settle(st_);
    return 1;
  }
  throw new Error('ukjend token «' + tok + '»');
}
function parsePosition(words) {
  var i = 1, s2;
  if (words[i] === 'startpos') { s2 = SS.fromFEN(START_FEN); i++; }
  else if (words[i] === 'fen') {
    var rest = words.slice(i + 1);
    var mIx = rest.indexOf('moves');
    var fen = mIx >= 0 ? rest.slice(0, mIx).join(' ') : rest.join(' ');
    s2 = SS.fromFEN(fen);
    i = mIx >= 0 ? i + 1 + mIx + 1 : words.length;
  } else throw new Error('position: startpos|fen');
  // Regelmodus: FEN-en ber ikkje Nesseby-flaget, so det kjem frå setoption.
  // (I appen vert motoren kalla med stillinga direkte, og då er st.nesseby alt sett.)
  if (NESSBY) s2.nesseby = true;
  if (words[i] === 'moves') i++;
  while (i < words.length) i += applyToken(s2, words[i], words, i);
  return s2;
}
function valueOf(st_) {
  // verd frå sida-som-skal-trekke sitt syn
  if (st_.winner !== null && st_.winner !== undefined) return st_.winner === st_.turn ? 1 : -1;
  var e = encodeState(st_);
  return netForward(e.planes, e.glob).value;
}
// Eitt trekk framover med verdinettet (valfritt, option Lookahead=k): dei k
// beste policy-trekka vert spela på ein KLON av den ekte tilstanden, og
// barnestillinga vert verdsett av verdinettet (med forteikn for kven som
// trekker). Politikk-hovudet er trena på utfallsvekta mål, men er svakt åleine;
// verdinettet ber informasjonen som ikkje hamnar i argument-maks.
// Kan motstandaren slå den brikka vi netto flytta, med ein av terningverdiane
// 1-3? (Bruka berre i materialsklausa; billig: éin trekk-generering per verdi.)
function opponentCanTakeBack(ch, mv) {
  var p = SS.piece(ch, mv.pid);
  if (!p || p.captured) return true;
  var sq = p.t === 's' ? SS.soldierAt(p.o, p.at) : p.at;
  var save = { dice: ch.dice ? ch.dice.slice() : null,
               used: ch.used ? ch.used.slice() : null, phase: ch.phase, turn: ch.turn };
  var taken = false;
  try {
    ch.phase = 'move';
    var faces = [['X', 1], ['II', 2], ['III', 3]];
    for (var i = 0; i < 3 && !taken; i++) {
      ch.dice = [faces[i][0], '_', '_'];
      ch.used = [false, true, true];
      var pool = SS.legalMoves(ch, 1 - p.o);        // p.o = sida som netto flytta
      for (var j = 0; j < pool.length; j++) {
        if (SS.moveDest(ch, pool[j]) === sq && pool[j].pid !== mv.pid) { taken = true; break; }
      }
    }
  } finally {
    ch.dice = save.dice; ch.used = save.used; ch.phase = save.phase; ch.turn = save.turn;
  }
  return taken;
}

function childValue(ch, rootTurn) {
  if (ch.winner !== null && ch.winner !== undefined) return ch.winner === rootTurn ? 1 : -1;
  var v = valueOf(ch);
  return ch.turn === rootTurn ? v : -v;
}
// 2-ply: etter vårt trekk er motstandaren i tur. Verdistett vårt trekk som
// det VERSTE motstandaren kan gjere mot oss, blant dei m beste svara hans
// (etter policy). Det fangar t.d. «trekket mitt ser bra ut, men han tek
// dronninga mi neste gong» — noko 1-ply lookahead ikkje ser.
function deepScore(ch, rootTurn, m) {
  var ids = legalIds(ch);
  if (!ids.length) return childValue(ch, rootTurn);
  var real = [];
  for (var i = 0; i < ids.length; i++) if (ids[i][0] < SPEC_BASE) real.push(ids[i]);
  if (!real.length) return childValue(ch, rootTurn);   // berre pass/reroll
  var e = encodeState(ch), f = netForward(e.planes, e.glob);
  real.sort(function (a, b) { return f.policy[b[0]] - f.policy[a[0]]; });
  var worst = Infinity;
  for (var j = 0; j < Math.min(m, real.length); j++) {
    var tok = encode(ch, real[j][1]);
    if (!tok) continue;
    var g = SS.deserialize(SS.serialize(ch));
    var mv2 = decode(g, tok);
    if (!mv2) continue;
    SS.applyMove(g, mv2); SS.settle(g);
    var v = childValue(g, rootTurn);
    if (v < worst) worst = v;
  }
  return worst === Infinity ? childValue(ch, rootTurn) : worst;
}

function pick(st_) {
  var ids = legalIds(st_);
  var enc = encodeState(st_);
  var r = netForward(enc.planes, enc.glob);
  var best = -Infinity, bestA = -1;
  for (var i = 0; i < ids.length; i++) {
    var v = r.policy[ids[i][0]];
    if (v > best) { best = v; bestA = ids[i][0]; }
  }
  // berre verkelege trekk kan evaluerast (pass/reroll er ikkje stillingar)
  var real = [];
  for (var j = 0; j < ids.length; j++) if (ids[j][0] < SPEC_BASE) real.push(ids[j]);
  if (LA > 0 && real.length > 1) {
    real.sort(function (a, b) { return r.policy[b[0]] - r.policy[a[0]]; });
    var cand = real.slice(0, Math.min(LA + 1, real.length));
    // SLAG SKAL ALLTID VURDERAST, ikkje berre om dei er mellom dei k beste
    // etter policy. Nettet er trena på data der slag var sjeldne, so eit
    // vinnande slag kan hamne på 12. plass og aldri verte rekna ut.
    // (Lars 2026-10-27: AI-en oversåg konge-/dronningkapslar.)
    var inCand = {};
    for (var ci = 0; ci < cand.length; ci++) inCand[cand[ci][0]] = 1;
    for (var ri = 0; ri < real.length; ri++) {
      var mvR = real[ri][1];
      var dR = SS.moveDest(st_, mvR);
      var onR = SS.occ ? SS.occ(st_, dR) : null;
      var kill = false;
      if (onR) {
        for (var oi = 0; oi < onR.length; oi++) {
          var q = onR[oi];
          if (q.o != null && q.o !== st_.turn && q.id !== mvR.pid) kill = true;
        }
      }
      if (kill && !inCand[real[ri][0]]) { cand.push(real[ri]); inCand[real[ri][0]] = 1; }
    }
    var bestV = -Infinity, bestC = cand[0][0];
    // MATERIALSKLAUSE (quiescence-light). Verdhovudet er eit tilnærming: målt
    // på eit verkeleg parti undervurderte det eit GRATIS slag med 0,08 (nettet
    // meinte dronninga stod dårleg på E2, men ho overlevde 27 trekk i utspel
    // og gav eit soldatkast). Materiale er konkret, verdiestimatet er ikkje:
    // er eit trygt slag (ingen motslag) innanfor CAP_MARGIN av beste trekk,
    // vel vi slaget.
    var bestCapV = -Infinity, bestCapA = -1;
    for (var c = 0; c < cand.length; c++) {
      var tok = encode(st_, cand[c][1]);
      var ch = SS.deserialize(SS.serialize(st_));
      var mv2 = decode(ch, tok);
      if (!mv2) continue;
      SS.applyMove(ch, mv2); SS.settle(ch);
      var cv = (DEEP > 0 && ch.winner == null)
        ? deepScore(ch, st_.turn, DEEP)
        : childValue(ch, st_.turn);
      if (cv > bestV) { bestV = cv; bestC = cand[c][0]; }
      // trygt slag?  tok noko, og kan motstandaren ikkje ta tilbake?
      if (CAP_MARGIN > 0 && ch.winner == null &&
          ch.kills[st_.turn] > st_.kills[st_.turn] &&
          !opponentCanTakeBack(ch, mv2)) {
        if (cv > bestCapV) { bestCapV = cv; bestCapA = cand[c][0]; }
      }
    }
    if (bestCapA >= 0 && bestCapV >= bestV - CAP_MARGIN) {
      return { aid: bestCapA, value: bestCapV };
    }
    if (bestV > -Infinity) return { aid: bestC, value: bestV };
  }
  return { aid: bestA, value: r.value };
}
function doGo() {
  if (st.winner !== null || st.phase === 'over') { out('bestmove none'); return; }
  if (st.phase !== 'move' || !st.dice.length) {
    out('error go utan terningar (send diceroll)'); out('bestmove none'); return;
  }
  poolStrings(st).forEach(function (s) { out('info movepool ' + s); });
  var ids = legalIds(st);
  if (!ids.length) { out('bestmove ' + (SS.canRerollX(st) ? 'reroll' : 'pass')); return; }
  var pk = pick(st), tok;
  if (pk.aid === SPEC_BASE) tok = 'pass';
  else if (pk.aid === SPEC_BASE + 1) tok = 'reroll';
  else {
    var mv = null;
    for (var i = 0; i < ids.length; i++) if (ids[i][0] === pk.aid) mv = ids[i][1];
    tok = mv ? encode(st, mv) : 'pass';
  }
  if (dbg) out('info string value=' + pk.value.toFixed(3));
  out('bestmove ' + tok);
}
function out(s) { M._out(s); }

M._out = function () {};
M._quit = function () {};
M.onLine = function (line) {
  var words = line.trim().split(/\s+/);
  if (!words[0]) return;
  var cmd = words[0];
  try {
    if (cmd === 'uci') {
      out('id name ' + ENGINE_NAME);
      out('id author sahkku-ai (PyTorch -> JS export)');
      out('option name Debug type check default false');
      out('option name Lookahead type spin default 4 min 0 max 12');
      out('option name Deep type spin default 0 min 0 max 4');
      out('option name CapMargin type string default 0.12');
      out('option name Nesseby type check default false');
      out('uciok');
    } else if (cmd === 'isready') {
      out('readyok');
    } else if (cmd === 'ucinewgame') {
      st = SS.fromFEN(START_FEN); ilStreak = 0;
    } else if (cmd === 'setoption') {
      var nIx = words.indexOf('name'), vIx = words.indexOf('value');
      if (nIx >= 0 && vIx >= 0) {
        var onm = words.slice(nIx + 1, vIx).join('').toLowerCase();
        var oval = words.slice(vIx + 1).join('');
        if (onm === 'debug') dbg = oval === 'true';
        else if (onm === 'lookahead') {
          var n = parseInt(oval, 10);
          LA = isNaN(n) ? 4 : Math.max(0, Math.min(12, n));
        }
        else if (onm === 'nesseby') {
          NESSBY = (oval === 'true' || oval === '1');
        }
        else if (onm === 'capmargin') {
          var cm = parseFloat(oval);
          CAP_MARGIN = isNaN(cm) ? 0.12 : Math.max(0, Math.min(1, cm));
        }
        else if (onm === 'deep') {
          var d = parseInt(oval, 10);
          DEEP = isNaN(d) ? 0 : Math.max(0, Math.min(4, d));
        }
      }
    } else if (cmd === 'position') {
      st = parsePosition(words); ilStreak = 0;
    } else if (cmd === 'diceroll') {
      if (words.length < 4) { out('error diceroll treng 3 terningar'); return; }
      var faces = [words[1], words[2], words[3]].map(function (f) {
        if (!DEC[f]) throw new Error('ulovleg terning «' + f + '»');
        return DEC[f];
      });
      st.dice = faces;
      st.used = faces.map(function (d) { return d === '_'; });
      st.diceBy = st.turn;
      st.phase = 'move';
    } else if (cmd === 'go' || cmd === 'getmove') {
      doGo();
    } else if (cmd === 'movelist') {
      poolStrings(st).forEach(function (s) { out('info movepool ' + s); });
      out('movepool done');
    } else if (cmd === 'debug') {
      dbg = words[1] === 'on';
    } else if (cmd === 'stop') {
      /* momentmotor: inga søk å avbryte */
    } else if (cmd === 'save' || cmd === 'load') {
      out('error unsupported');
    } else if (cmd === 'quit') {
      M._quit();
    } else {
      out('error ukjend kommando «' + cmd + '»');
    }
  } catch (e) {
    ilStreak++;
    out('error ' + (e && e.message ? e.message : e));
    if (ilStreak > 3) out('bestmove none');
  }
};
M.exported = {
  SS: SS, loadWeights: loadWeights, netForward: netForward,
  encodeState: encodeState, legalIds: legalIds, moveId: moveId,
  state: function () { return st; },
  // ---- nettlesar-API (SnartSahkku sin motorveljar) ----
  // Gjev trekkobjektet til den beste aksjonen for ein vilkårleg tilstand, i
  // same form som SS.legalMoves: {pid, dir, v, kind}. pass/reroll gjev null,
  // slik at GUI-en held fram med sin eigen settle-flyt.
  chooseMove: function (st_) {
    var ids = legalIds(st_);
    if (!ids.length) return null;
    var pk = pick(st_);
    for (var i = 0; i < ids.length; i++) {
      if (ids[i][0] === pk.aid) {
        var mv = ids[i][1];
        return (mv && mv.pid !== undefined) ? mv : null;
      }
    }
    return null;
  },
  // Vel ei HANDLING (ikkje berre eit trekk): omkast og pass er no val, og
  // GUI-et skal ikkje kaste om på eigen hand. Returnerer {kind, aid, mv?}.
  chooseAction: function (st_) {
    var ids = legalIds(st_);
    if (!ids.length) return null;
    var pk = pick(st_);
    for (var i = 0; i < ids.length; i++) {
      if (ids[i][0] !== pk.aid) continue;
      var mv = ids[i][1], k = mv && mv.kind ? mv.kind : 'mv';
      return { kind: (k === 'pass' || k === 'reroll') ? k : 'move',
               aid: ids[i][0], mv: mv };
    }
    return null;
  },
  setLookahead: function (k) {
    var n = parseInt(k, 10);
    LA = isNaN(n) ? 4 : Math.max(0, Math.min(12, n));
  }
};
return M;
})();
if (typeof require !== 'undefined' && typeof module !== 'undefined') {
  var fs = require('fs'), path = require('path');
  NN.exported.loadWeights(JSON.parse(fs.readFileSync(path.join(__dirname, '__WEIGHTS_FILE__'), 'utf8')));
  if (process.argv[2] === 'features-test') {
    // stdin: JSON [{fen, dice?, used?}…]; stdout: JSON planes/policy-at-legal/value
    var inp = JSON.parse(fs.readFileSync(0, 'utf8'));
    var res = inp.map(function (q) {
      var stq = NN.exported.SS.fromFEN(q.fen);
      var enc = NN.exported.encodeState(stq);
      var ids = NN.exported.legalIds(stq);
      var r = NN.exported.netForward(enc.planes, enc.glob);
      return {
        fen: q.fen,
        planes: Array.from(enc.planes).map(function (v) { return Math.round(v * 1e5) / 1e5; }),
        glob: Array.from(enc.glob).map(function (v) { return Math.round(v * 1e5) / 1e5; }),
        legal: ids.map(function (x) { return x[0]; }),
        policy: ids.map(function (x) { return Math.round(r.policy[x[0]] * 1e5) / 1e5; }),
        value: Math.round(r.value * 1e5) / 1e5
      };
    });
    process.stdout.write(JSON.stringify(res));
    process.exit(0);
  }
  var rl = require('readline').createInterface({ input: process.stdin, terminal: false });
  NN._out = function (s) { process.stdout.write(s + '\n'); };
  NN._quit = function () { process.exit(0); };
  rl.on('line', function (l) { NN.onLine(l); });
}
