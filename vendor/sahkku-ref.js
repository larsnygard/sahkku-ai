#!/usr/bin/env node
/* sahkku-ref — S-UCI 1.0 referansje-motor for sáhkku (spec: docs/SAHKKU-UCI.md).
 * Snakkar S-UCI paa stdin/stdout; spelreglane ligg i js/engine.js (dommar),
 * speler-valet i js/cpu.js (nivaa 1-3). Ingen tilfeldige tal: terningane
 * kjem fra GUI-en via `diceroll`.
 */
'use strict';
const readline = require('readline');

global.SS = require('./engine.js');
require('./fen.js');                       // SS.toFEN / fromFEN / sqOfName
const CPU = require('./cpu.js');
const SS = global.SS;

const DEC = { X: 'X', '2': 'II', '3': 'III', '_': '_' };

let optLevel = 2;
let optNesseby = false;          // regelmodus: false = klassisk, true = Nesseby
let dbg = false;

// ---------- trekk-koding: {pid,dir,v,kind} -> "A1-B1" ----------
function encode(st, mv) {
  const p = SS.piece(st, mv.pid);
  if (p.t === 's') {
    const from = SS.soldierAt(p.o, p.at);
    const to = SS.soldierAt(p.o, p.at + (mv.kind === 'rel' ? 1 : mv.v));
    return SS.name(from) + '-' + SS.name(to);
  }
  if (mv.to !== undefined && mv.to !== null) return SS.name(p.at) + '-' + SS.name(mv.to);
  const r = SS.pathFor(st, p, mv.dir, mv.v);
  if (!r.legal || !r.path.length) return null;
  return SS.name(p.at) + '-' + SS.name(r.path[r.path.length - 1]);
}
function poolStrings(st) {
  const seen = new Set(), out = [];
  SS.legalMoves(st, st.turn).forEach(function (mv) {
    const s = encode(st, mv);
    if (s && !seen.has(s)) { seen.add(s); out.push(s); }
  });
  return out;
}
function decode(st, str) {
  const parts = str.split('-');
  if (parts.length !== 2) return null;
  const from = SS.sqOfName(parts[0]), to = SS.sqOfName(parts[1]);
  if (from < 0 || to < 0) return null;
  const lms = SS.legalMoves(st, st.turn);
  for (const mv of lms) if (encode(st, mv) === str) return mv;
  return null;
}

// ---------- posisjonsmaskin: replay av trekktokens (spec §5) ----------
function applyToken(st, tok, words, i) {
  if (tok === 'roll') {
    const faces = [words[i + 1], words[i + 2], words[i + 3]].map(function (f) {
      if (!DEC[f]) throw new Error('ulovleg terning «' + f + '»');
      return DEC[f];
    });
    st.dice = faces;
    st.used = faces.map(function (d) { return d === '_'; });
    st.diceBy = st.turn;
    st.phase = 'move';
    SS.logPly(st).roll.push(faces.slice());
    return 4;
  }
  if (tok === 'pass') { SS.passRemaining(st); SS.settle(st); return 1; }
  if (tok === 'reroll') { return 1; }
  if (tok === 'resign') {
    st.winner = 1 - st.turn; st.winReason = 'overgivelse'; st.phase = 'over';
    return 1;
  }
  if (/^[A-O][123]-[A-O][123]$/.test(tok)) {
    const mv = decode(st, tok);
    if (!mv) {
      console.error('DEBUG applyToken: ulovleg «' + tok + '» turn=' + st.turn + ' faza=' + st.phase + ' FEN=' + SS.toFEN(st));
      throw new Error('ulovleg trekk «' + tok + '»');
    }
    SS.applyMove(st, mv);
    SS.settle(st);
    return 1;
  }
  throw new Error('ukjend token «' + tok + '»');
}

function parsePosition(words) {
  // words: position startpos|fen <fen...> [moves t...]
  // applyToken returnerer TALEN PÅ TOKEN som er brukte (roll = 4, anna = 1)
  let i = 1, st;
  if (words[i] === 'startpos') { st = SS.fromFEN(START_FEN); i++; }
  else if (words[i] === 'fen') {
    const rest = words.slice(i + 1);
    let fen, mIx = rest.indexOf('moves');
    fen = mIx >= 0 ? rest.slice(0, mIx).join(' ') : rest.join(' ');
    st = SS.fromFEN(fen);
    i = mIx >= 0 ? i + 1 + mIx + 1 : words.length;
  } else throw new Error('position: startpos|fen');
  // Regelmodusen må inn FØR trekka vert brukte: same POSITION-token under
  // Nesseby og klassisk er ikkje same stilling.
  st.nesseby = optNesseby === true;
  if (words[i] === 'moves') i++;      // spec §5: "position startpos moves <t…>" er lovleg
  while (i < words.length) i += applyToken(st, words[i], words, i);
  return st;
}

const START_FEN = 'g-g-g-g-g-g-g-g-g-g-g-g-g-g-g/.-.-.-d-.-.-.-N-.-.-.-q-.-.-./m-m-m-m-m-m-m-m-m-m-m-m-m-m-m w roll - 000 H2 0:0 0 *';
let st = SS.fromFEN(START_FEN);

function out(s) { process.stdout.write(s + '\n'); }
function loginfo(s) { if (dbg) out('info string ' + s); }

// ---------- go ----------
function doGo(words) {
  if (st.winner != null || st.phase === 'over') { out('bestmove none'); return; }
  if (st.phase !== 'move' || !st.dice.length) { out('error go utan terningar (send diceroll)'); out('bestmove none'); return; }
  poolStrings(st).forEach(function (s) { out('info movepool ' + s); });
  const rk = SS.activeRank(st);
  if (rk == null) { out('bestmove pass'); return; }
  if (!SS.legalMoves(st, st.turn).length) {
    out('bestmove ' + (SS.canRerollX(st) ? 'reroll' : 'pass'));
    return;
  }
  const mv = CPU.pick(st, optLevel);
  const s = mv ? encode(st, mv) : null;
  if (!s) { out('bestmove pass'); return; }
  if (dbg) console.error('DEBUG go: turn=' + st.turn + ' dice=' + st.dice.join(',') +
    ' pool=' + poolStrings(st).join(',') + ' pick=' + s);
  out('bestmove ' + s);
}

// ---------- hovudløkke ----------
let illegalStreak = 0;
const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', function (line) {
  const words = line.trim().split(/\s+/);
  if (!words[0]) return;
  const cmd = words[0];
  try {
    if (cmd === 'uci') {
      out('id name SahkkuRef 1.0');
      out('id author SnartSahkku (reference)');
      out('option name Level type spin default 2 min 1 max 3');
      out('option name Nesseby type check default false');
      out('uciok');
    } else if (cmd === 'isready') {
      out('readyok');
    } else if (cmd === 'ucinewgame') {
      st = SS.fromFEN(START_FEN);
      st.nesseby = optNesseby === true;
      illegalStreak = 0;
    } else if (cmd === 'setoption') {
      const nameIx = words.indexOf('name');
      const name = nameIx >= 0 ? words[nameIx + 1] : '';
      const vIx = words.indexOf('value');
      if (name === 'Level' && vIx >= 0) {
        const v = parseInt(words[vIx + 1], 10);
        if (v >= 1 && v <= 3) optLevel = v; else out('error Level 1-3');
      } else if (name === 'Nesseby' && vIx >= 0) {
        const s = String(words[vIx + 1] || '').toLowerCase();
        optNesseby = (s === 'true' || s === '1' || s === 'on' || s === 'yes');
        if (st) st.nesseby = optNesseby;   // gjelda òg ei alt oppsett stilling
      } // ukjende namn: ignorer (spec §2)
    } else if (cmd === 'position') {
      st = parsePosition(words);
      illegalStreak = 0;
      loginfo('position -> ' + SS.toFEN(st));
    } else if (cmd === 'diceroll') {
      if (words.length < 4) { out('error diceroll treng 3 terningar'); return; }
      const faces = [words[1], words[2], words[3]].map(function (f) {
        if (!DEC[f]) throw new Error('ulovleg terning «' + f + '»');
        return DEC[f];
      });
      st.dice = faces;
      st.used = faces.map(function (d) { return d === '_'; });
      st.diceBy = st.turn;
      st.phase = 'move';
      SS.logPly(st).roll.push(faces.slice());
      loginfo('diceroll -> ' + SS.toFEN(st));
    } else if (cmd === 'go') {
      doGo(words);
    } else if (cmd === 'movelist') {
      poolStrings(st).forEach(function (s) { out('info movepool ' + s); });
      out('movepool done');
    } else if (cmd === 'stop') {
      // momentmotor: inga søk å avbryte
    } else if (cmd === 'debug') {
      dbg = words[1] === 'on';
    } else if (cmd === 'save' || cmd === 'load') {
      out('error unsupported');
    } else if (cmd === 'quit') {
      rl.close();
      process.exit(0);
    } else if (cmd === 'getmove') {
      // GUI-oppfolging etter ulovleg svar: same go ein gong til
      doGo(words);
    } else {
      out('error ukjend kommando «' + cmd + '»');
    }
  } catch (e) {
    out('error ' + e.message);
  }
});
