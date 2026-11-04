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
  // g[16]/g[17] = bonderisiko: kor stor del av kvar side sine utløyste soldatar
  // som står der motstandaren kan slå dei i neste trekk. `caps` er alt rekna ut
  // for fienden; vi treng same tabell for mi eiga side.
  var usMe = usableAndCapture(st, me);
  var expMe = 0, expFoe = 0;
  for (var si = 0; si < st.pieces.length; si++) {
    var q = st.pieces[si];
    if (q.t !== 's' || !q.released || q.captured) continue;
    var sq = SS.soldierAt(q.o, q.at);
    if (q.o === me) { if (us.caps[sq]) expMe++; }
    else { if (usMe.caps[sq]) expFoe++; }
  }
  g[16] = expMe / 15;
  g[17] = expFoe / 15;

  // ---- KAISER-0: syklusposisjon, slott-avstand, terningar, Nesseby-port
  // (spegl av features.py g[18..26]; alt rekt ut frå stillingsdata åleine
  //  — pass-streak og andre felt utanfor referee-en høyrer ikkje heime.)
  // g[18]/g[19] = snitt syklusposisjon (p.at, 0..59) for utløyste soldatar.
  function meanPos(side) {
    var ps = SS.aliveSoldiers(st, side).filter(function (p) { return p.released; });
    if (!ps.length) return 0;
    var s = 0;
    for (var mi = 0; mi < ps.length; mi++) s += ps[mi].at / 60;
    return s / ps.length;
  }
  g[18] = meanPos(me);
  g[19] = meanPos(foe);
  // g[20] = kongen sin avstand frå slottet H2, snudd (1.0 = heime).
  if (k) g[20] = 1 - Math.min(cheb(k.at, 22), 14) / 14;
  // g[21] = dronninga mi sin avstand frå heimfeltet (L2=18 / D2=26), snudd.
  if (qm) g[21] = 1 - Math.min(cheb(qm.at, me === 0 ? 18 : 26), 14) / 14;
  // g[22] = ubrukte X-terningar i kastet /3.
  var xLeft = 0;
  for (var di = 0; di < Math.min(3, (st.dice || []).length); di++) {
    if (!st.used[di] && st.dice[di] === 'X') xLeft++;
  }
  g[22] = xLeft / 3;
  // g[23] = aktiv terningverdi /3.
  g[23] = (v || 0) / 3;
  // g[24] = brukbare terningar att i handa /3 (dice_left — same tal i
  //  begge refereear; pass-streak finst ikkje i JS og kan ikkje vere trekk).
  g[24] = (SS.diceLeft ? SS.diceLeft(st)
                       : (st.dice || []).filter(function (d, i2) { return !st.used[i2] && d !== '_'; }).length) / 3;
  // g[25]/g[26] = NESSEBY-NØKLELEN: porten for landingskap av kongen.
  //  g[25] = mine soldatar på FIENDEN sin heimrad; g[26] = fienden sine
  //  soldatar på MI heimrad. (rad: menn heime = 0, koner heime = 2.)
  var gateMe = 0, gateFoe = 0;
  var foeRow = (me === 0) ? 2 : 0;        // fienden sin heimrad
  var myRow = (me === 0) ? 0 : 2;         // mi heimrad
  for (var gi = 0; gi < st.pieces.length; gi++) {
    var qp = st.pieces[gi];
    if (qp.t !== 's' || !qp.released || qp.captured) continue;
    var gsq = SS.soldierAt(qp.o, qp.at), grow = SS.pt(gsq).row;
    if (qp.o === me) { if (grow === foeRow) gateMe++; }
    else { if (grow === myRow) gateFoe++; }
  }
  g[25] = gateMe / 15;
  g[26] = gateFoe / 15;
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
var REROLL_EVAL = 0;     // 0 = gamal oppførsel (reroll berre når tvungen)
                         // 1 = 64-rollsforventning samanlikna med trekka
                         // 2 = som 1, men krev REROLL_MARGIN over beste trekk
var REROLL_MARGIN = 0.0; // margin som krevst for å omkaste framfor beste trekk
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
  // Materialblending (KAISER.1): stillingsnettet pluss justerbart materiale.
  // Standard blend=0: NAIS.9-oppførselen er uendra; KAISER slår det på med
  // `setoption name Material value ...`. blend=0 gjev nøyaktig gamal kodeveg
  // (materiale reknast ikkje ein gong), so alt som er validert med NAIS.9
  // held seg til KAISER er godteken.
  if (MAT.blend !== 0) v = clamp1(v + MAT.blend * materialSum(ch));
  return ch.turn === rootTurn ? v : -v;
}
// ---- MATERIELLE VERDIAR (spegl av sahkku_ai/materials.py) ----------------
// Justerbare i sanntid: `setoption name Material value qActPerS=-0.05,...`
// POV-DELTT: våre brikker +, deira − (FORLELESGNAD, ikkje invent-list).
// Dronninga sin pris er SKALERT etters Lars sine reglar: fleire eigne
// soldatar, fleire brikker på brettet, og kongen borte frå slottet gjer
// aktiveringa stadig farlegare. FIENDEN sin utløyste dronning er GEVINST.
var MAT_DEFAULT = {
  sHome: 0.02, sAct: 0.05, qHome: 0.05,
  qActBase: -0.08, qActPerS: -0.035, qActDen: -0.005, qActNoKing: -0.12,
  qActEnemy: 0.14, king: 0.04, kFar: -0.03, blend: 0.30
};
var MAT = /*§MATERIAL§*/MAT_DEFAULT;
var KING_NEAR = 4;
function materialSum(st_) {
  var me = st_.turn, s = 0, mine = 0, n = 0, kp = null;
  var i, p;
  for (i = 0; i < st_.pieces.length; i++) {
    p = st_.pieces[i];
    if (p.captured) continue;
    n++;
    if (p.t === 's' && p.o === me) mine++;
    if (p.t === 'k' && p.o === me) kp = p;   // berre MIN konge vernar dronninga
  }
  // Lars: «viss du ikkje kontrollerer kongen, er det nesten umogleg å verne
  // dronninga» — ANTAN manglande eigen konge, eller kongen langt frå slottet
  var kingFar = !kp || Math.min(cheb(kp.at, SS.CASTLE), 14) > KING_NEAR;
  for (i = 0; i < st_.pieces.length; i++) {
    p = st_.pieces[i];
    if (p.captured || p.o === null) continue;   // nøytral konge tel ikkje
    var sign = p.o === me ? 1 : -1;
    if (p.t === 's') s += sign * (p.released ? MAT.sAct : MAT.sHome);
    else if (p.t === 'q') {
      if (!p.released) s += sign * MAT.qHome;
      else if (p.o === me) s += MAT.qActBase + MAT.qActPerS * mine + MAT.qActDen * n +
                             (kingFar ? MAT.qActNoKing : 0);
      else s += MAT.qActEnemy;
    }
    else if (p.t === 'k') s += sign * (MAT.king + MAT.kFar * (Math.min(cheb(p.at, SS.CASTLE), 14) / 14));
  }
  return s;
}
function clamp1(x) { return x < -1 ? -1 : (x > 1 ? 1 : x); }
function applyMatOption(oval) {
  var parts = oval.replace(/;/g, ',').split(',');
  for (var i = 0; i < parts.length; i++) {
    var kv = parts[i].split('=');
    if (kv.length !== 2) continue;
    var k = kv[0].trim();
    if (MAT[k] !== undefined) { var x = parseFloat(kv[1]); if (!isNaN(x)) MAT[k] = x; }
  }
}

function matStr() {
  return Object.keys(MAT_DEFAULT).map(function (k) { return k + '=' + MAT[k]; }).join(',');
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

// ---- REROLL-EVALUERING (KAISER.1): 64-rollsforventning ==================
// Ein omkast av eit X-terning er ein SANSLIGKnute: verdien er forventinga
// over det neste kastet. Terningane er 4-siders (X, II, III, blank) — 4^3 =
// 64 like sannsynlege utfall. For KVART utfall spelar vi det beste policy-
// svaret (djupt etter LA/DEEP) og tek verdien; gjennomsnittet er verdien av
// å omkaste. Sammenliknast med beste verkelege trekk i pick().
// 64 grunutfall dekkjer heile fordelinga (XII+XIII+XIII... er ikkje
// eigne utfall — kvar terning er X/II/III/blank), så dette er EKSAKT, ikkje
// sampling. Kostnad: 64 × (éin policy-evaluering + opp til LA barn) — berre
// på stillinger der X kan omkastast.
function rollFaces() {
  if (!rollFaces._f) {
    var f = [];
    for (var a = 0; a < 4; a++) for (var b = 0; b < 4; b++) for (var c = 0; c < 4; c++) {
      f.push([SS.DICE[a], SS.DICE[b], SS.DICE[c]]);
    }
    rollFaces._f = f;
  }
  return rollFaces._f;
}
function rerollValue(st_, root, depth) {
  // Verdi av å omkaste eit ubrukt X: 64 like sannsynlege fjes, beste
  // policy-svar eitt trekk djupt, memoert (utfalla går att på tvers av
  // heile treet). Utan røyr: omkasten er berre ein ny kastfase-forventning.
  var faces = rollFaces(), sum = 0;
  for (var i = 0; i < faces.length; i++) {
    var g = SS.deserialize(SS.serialize(st_));
    var slotIx = -1;
    for (var s = 0; s < 3; s++) if (g.dice[s] === 'X' && !g.used[s]) { slotIx = s; break; }
    if (slotIx < 0) return -Infinity;
    g.dice = g.dice.slice(); g.used = g.used.slice();
    g.dice[slotIx] = faces[i][0];
    g.used[slotIx] = (faces[i][0] === '_');
    // Dei andre terningane står URANDA (rerollX byter berre X-feltet), og
    // allereie brukte felt verer BLI brukte.
    g.phase = 'move';
    var ids = legalIds(g);
    var real = [];
    for (var j = 0; j < ids.length; j++) if (ids[j][0] < SPEC_BASE) real.push(ids[j]);
    var v;
    if (!real.length) {
      // inga røyr med dette fjeset: omkast = ny kastfase for same side
      var gr = SS.deserialize(SS.serialize(g));
      gr.dice = []; gr.used = []; gr.phase = 'roll';
      v = (depth <= 0 || gr.winner != null) ? leafRoot(gr, root)
                                            : rollPhaseValue(gr, root, depth, false);
    } else {
      var fr = memoEval(g);
      real.sort(function (a, b) { return fr.policy[b[0]] - fr.policy[a[0]]; });
      var top = real.slice(0, Math.min(3, real.length));
      var best = -Infinity;
      for (var t = 0; t < top.length; t++) {
        var tok = encode(g, top[t][1]);
        if (!tok) continue;
        var h = SS.deserialize(SS.serialize(g));
        var mv = decode(h, tok);
        if (!mv) continue;
        SS.applyMove(h, mv); SS.settle(h);
        var cv = (depth <= 1 || h.winner != null)
          ? leafRoot(h, root) : searchPos(h, root, depth - 1, -Infinity, Infinity);
        if (cv > best) best = cv;
      }
      v = best === -Infinity ? leafRoot(g, root) : best;
    }
    sum += v;
  }
  return sum / faces.length;
}
function evalReroll(st_, root, depth) {
  // spelarkommandoen (go) brukar depth=0; scorefen-søket tek med djupet sitt
  return rerollValue(st_, root, depth === undefined ? 0 : depth);
}

// ---- SCOREFEN-SØK (KAISER.1): alpha-beta med policy-ordning =============
// scoreSearch(st, root, depth, alfa, beta): verdi frå `root` si side.
//  * Blad (depth=0): netten (+materiale), forteikn for kven som søkjer.
//  * Trekkfase: verkelege trekk, policy-ordna, breidd W=6; alpha-beta-
//    klipping. Reroll av X er ein SANSLIGKnute med verdi = 64-forventning
//    (rerollValue) og tek del i MAX-leddet som alle andre trekk.
//  * Kastfase (ingen terning kasta): forventa verdi = gjennomsnitt over
//    64 kva-kast-utfall (same utfallsrom som terningane). Kvart utfall
//    searchast vidare med eitt trekk djupare.
//  * Pass (inga røyr): kasta vert avslutta, turden byter; verdien er
//    stillinga etter passet søkt med djupde-1.
// FART (KAISER.1): netForward er ~100 ms i rein JS (~22M MACs). Full breidd
//  av 64-utslaget ved kvar knute eksploderer (64² per trekkpar). Konvensjon:
//  i KASTFASEN er verdien = gjennomsnitt over 64 utslag av BESTE POLICY-SVAR
//  eitt trekk djupt (standard for terningspill: «best response»), ikkje
//  fullgjera søk under kvart utslag. Kvart utslag er memoert — dei same
//  stillingane går att på tvers av syskenknuter, så memoen gjer kastfasen
//  til ~64+64 netto-evalueringar PR SIDE-TUR (ikkje per trekk).
//  I TREKKFASA: policy-ordna breidd 6 med alpha-beta (EXACT/LOW/UPPER i TT).
//  NODE_TAK tel netto-evalueringar; ved takket vert bladet brukt og svaret
//  er ei grense (oppgitt som `cap=1` i info-linja).
var MEMO = new Map(), MEMO_CAP = 4000;
var TT = new Map(), TT_CAP = 60000;
var NODES = 0, NODE_TAK = 4000;
function memoEval(st_) {
  var key = SS.serialize(st_);
  var hit = MEMO.get(key);
  if (hit !== undefined) { MEMO.delete(key); MEMO.set(key, hit); return hit; }
  var e = encodeState(st_);
  var r = netForward(e.planes, e.glob);
  hit = { enc: e, policy: r.policy, value: r.value };
  NODES++;
  MEMO.set(key, hit);
  if (MEMO.size > MEMO_CAP) { var it = MEMO.keys().next(); if (!it.done) MEMO.delete(it.value); }
  return hit;
}
function blended(st_, rawValue) {
  return MAT.blend !== 0 ? clamp1(rawValue + MAT.blend * materialSum(st_)) : rawValue;
}
function leafRoot(ch, root) {
  // bladverdi frå rota si side (netten tel frå sida-som-trekkjer)
  var v = blended(ch, memoEval(ch).value);
  return ch.turn === root ? v : -v;
}
var SEARCH_W = 6;

// Kastfase: 64-utslagsforventning over beste policy-svar (memoert per utslag).
// rerollOk=false: kastet er ALLEREIE omkasta (éin omkast per kast) — då
// vert ikkje omkastet tilbodt att (lèt òg som stoppar gjensidig reskursjon).
function rollPhaseValue(ch, root, depth, rerollOk) {
  var faces = rollFaces(), sum = 0;
  for (var i = 0; i < faces.length; i++) {
    var g = SS.deserialize(SS.serialize(ch));
    g.dice = faces[i].slice();
    g.used = g.dice.map(function (d) { return d === '_'; });
    g.diceBy = g.turn;
    g.phase = 'move';
    var ids = legalIds(g);
    var real = [];
    for (var j = 0; j < ids.length; j++) if (ids[j][0] < SPEC_BASE) real.push(ids[j]);
    var v;
    if (!real.length) {
      if (rerollOk !== false && SS.canRerollX(g)) {
        v = rerollValue(g, root, depth);         // omkast-innslaget i kastfasen
      } else {
        var gp = SS.deserialize(SS.serialize(g));
        SS.passRemaining(gp);
        v = (depth <= 1 || gp.winner != null) ? leafRoot(gp, root)
                                              : searchPos(gp, root, depth - 1, -Infinity, Infinity);
      }
    } else {
      var fr = memoEval(g);
      real.sort(function (a, b) { return fr.policy[b[0]] - fr.policy[a[0]]; });
      var tok = encode(g, real[0][1]);
      var h = SS.deserialize(SS.serialize(g));
      var mv = tok ? decode(h, tok) : null;
      if (!mv) { v = leafRoot(g, root); }
      else {
        SS.applyMove(h, mv); SS.settle(h);
        v = (depth <= 1 || h.winner != null) ? leafRoot(h, root)
                                             : searchPos(h, root, depth - 1, -Infinity, Infinity);
      }
    }
    sum += v;
  }
  return sum / faces.length;
}

// Trekkfase: policy-ordna alpha-beta, breidd W; reroll som MAX-opning.
function searchMoves(ch, root, depth, alfa, beta) {
  var key = SS.serialize(ch) + '|' + depth;
  var tt = TT.get(key);
  if (tt !== undefined) {
    if (tt.f === 0) return tt.v;
    if (tt.f === 1 && tt.v >= beta) return tt.v;    // LOW ( Cause-clipped
    if (tt.f === 2 && tt.v <= alfa) return tt.v;    // UPPER
    alfa = Math.max(alfa, tt.v); beta = Math.min(beta, tt.v);
  }
  var ids = legalIds(ch);
  var real = [];
  for (var i = 0; i < ids.length; i++) if (ids[i][0] < SPEC_BASE) real.push(ids[i]);
  var fr = memoEval(ch);
  if (real.length) {
    real.sort(function (a, b) { return fr.policy[b[0]] - fr.policy[a[0]]; });
    real = real.slice(0, SEARCH_W);
  }
  var isMax = (ch.turn === root);
  var best = isMax ? -Infinity : Infinity;
  var a = alfa, b = beta;
  for (var r = 0; r < real.length; r++) {
    if (NODES >= NODE_TAK) break;
    var tok = encode(ch, real[r][1]);
    if (!tok) continue;
    var g = SS.deserialize(SS.serialize(ch));
    var mv = decode(g, tok);
    if (!mv) continue;
    SS.applyMove(g, mv); SS.settle(g);
    var v = searchPos(g, root, depth - 1, a, b);
    if (isMax) { if (v > best) best = v; if (best > a) a = best; }
    else { if (v < best) best = v; if (best < b) b = best; }
    if (a >= b) break;                                     // alpha-beta-klipping
  }
  // REROLL: MAX-spelaren sin rett (kastet er hans) — tel i MAX-leddet.
  if (isMax && ch.phase === 'move' && SS.canRerollX(ch) && NODES < NODE_TAK) {
    var rv = rerollValue(ch, root, depth);
    if (rv > best) best = rv;
    if (best > a) a = best;
  }
  if (!isFinite(best)) {
    // ingen røyr, ingen omkast: kasta avsluttast, turden går over
    var g2 = SS.deserialize(SS.serialize(ch));
    SS.passRemaining(g2);
    best = searchPos(g2, root, depth - 1, alfa, beta);
  }
  if (TT.size < TT_CAP) {
    var f = (best <= alfa) ? 2 : (best >= beta) ? 1 : 0;
    TT.set(key, { v: best, f: f });
  }
  return best;
}
function searchPos(ch, root, depth, alfa, beta) {
  if (ch.winner !== null && ch.winner !== undefined) return ch.winner === root ? 1 : -1;
  if (depth <= 0 || NODES >= NODE_TAK) return leafRoot(ch, root);
  if (ch.phase === 'roll' && !(ch.dice && ch.dice.length)) {
    var v = rollPhaseValue(ch, root, depth, true);
    if (TT.size < TT_CAP) TT.set(SS.serialize(ch) + '|' + depth, { v: v, f: 0 });
    return v;
  }
  return searchMoves(ch, root, depth, alfa, beta);
}
function scoreSearch(stF, root, depth, nodeTak) {
  MEMO.clear(); TT.clear(); NODES = 0;
  NODE_TAK = Math.max(50, nodeTak | 0);
  return searchPos(stF, root, Math.max(1, depth), -Infinity, Infinity);
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
  // REROLL-EVALUERING i partispel (KAISER.1): same 64-forventning som i
  //  scorefen, samanlikna med det valde trekket sin verdi (frå pick).
  //  Krev marginen REROLL_MARGIN; RerollEval=0 lèt heilt vere (NAIS-vegen).
  if (REROLL_EVAL > 0 && tok !== 'reroll' && tok !== 'pass' && SS.canRerollX(st)) {
    var rv = evalReroll(st, st.turn);
    if (rv > pk.value + REROLL_MARGIN) tok = 'reroll';
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
      out('option name RerollEval type spin default 0 min 0 max 2');
      out('option name RerollMargin type string default 0.0');
      out('option name Material type string default ' + matStr());
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
        else if (onm === 'rerolleval') {
          var re = parseInt(oval, 10);
          REROLL_EVAL = isNaN(re) ? 0 : Math.max(0, Math.min(2, re));
        }
        else if (onm === 'rerollmargin') {
          var rm = parseFloat(oval);
          REROLL_MARGIN = isNaN(rm) ? 0 : Math.max(-1, Math.min(1, rm));
        }
        else if (onm === 'material') {
          applyMatOption(oval);
        }
      }
    } else if (cmd === 'position') {
      st = parsePosition(words); ilStreak = 0;
    } else if (cmd === 'scorefen') {
      // scorefen <FEN> [depth d] [nodes n] — stillingsverdie frå sida som
      //  skal trekke sitt syn. d=0: netten (+materiale). d>=1: alpha-beta
      //  med policy-ordna trekk (breidd 6) + 64-utslag i kastfasen.
      var fenWords = words.slice(1);
      var dIx = fenWords.indexOf('depth');
      var nIx = fenWords.indexOf('nodes');
      var dep = dIx >= 0 ? parseInt(fenWords[dIx + 1], 10) : 1;
      var tak = nIx >= 0 ? parseInt(fenWords[nIx + 1], 10) : 1200;
      if (isNaN(dep)) dep = 1;
      dep = Math.max(0, Math.min(8, dep));
      if (isNaN(tak)) tak = 1200;
      tak = Math.max(50, Math.min(20000, tak));
      if (dIx >= 0) fenWords.splice(dIx, 2);
      if ((nIx = fenWords.indexOf('nodes')) >= 0) fenWords.splice(nIx, 2);
      var stF = SS.fromFEN(fenWords.join(' '));
      if (NESSBY) stF.nesseby = true;
      var t0 = Date.now();
      var val = (dep === 0) ? leafRoot(stF, stF.turn)
                            : scoreSearch(stF, stF.turn, dep, tak);
      var capped = NODES >= tak;
      var mat = materialSum(stF);
      out('info score fen value=' + val.toFixed(4) + ' depth=' + dep +
          ' net=' + memoEval(stF).value.toFixed(4) + ' mat=' + mat.toFixed(4) +
          ' blend=' + MAT.blend.toFixed(3) + ' nodes=' + NODES +
          ' ms=' + (Date.now() - t0) + (capped ? ' cap=1' : ''));
      out('bestmove none');
    } else if (cmd === 'material') {
      out('info material ' + matStr());
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
  // ---- KAISER.1-verktøy for testar og analysar --------------------------
  //  verdi av omkast (64-forventning), eller -Infinity utan omkastmoglegheit.
  //  Merk: i fase 'move' med ubrukt X er omkastet òg lovleg (regel 1), så
  //  guarden må sjå på X-feltet, ikkje på canRerollX (som krev fase 'roll').
  evalReroll: function (st_, depth) {
    var hasUnusedX = false;
    for (var s = 0; s < (st_.dice ? 3 : 0); s++) {
      if (st_.dice[s] === 'X' && !st_.used[s]) hasUnusedX = true;
    }
    if (!hasUnusedX) return -Infinity;
    return evalReroll(st_, st_.turn, depth === undefined ? 0 : depth);
  },
  //  stillingsverdi (net+materiale) frå rota si side, valfritt søk
  scoreFen: function (fen, depth, nodes) {
    var stq = SS.fromFEN(fen);
    if (NESSBY) stq.nesseby = true;
    return (depth <= 0) ? leafRoot(stq, stq.turn)
                        : scoreSearch(stq, stq.turn, depth, nodes || 1200);
  },
  setMaterial: function (opt) { applyMatOption(String(opt)); return { MAT: MAT }; },
  materialSum: function (st_) { return materialSum(st_); },
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
