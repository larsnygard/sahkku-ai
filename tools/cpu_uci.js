// cpu_uci — S-UCI-skal kring den klassiske CPU-en i appen (js/cpu.js).
//
// Så agenten kan møte den SAME CPU-en som spelarane møter i snart.com/sahkku,
// i tools/eval_match.js og i benkar elles. Nivå 3 = FLOKK (dei sju reglane),
// 2 = gamal greedy, 1 = tilfeldig. Ingen nett, ingen pakkar utanom appen.
//
// Bruk:  node tools/cpu_uci.js [--level 3]
//        (S-UCI på stdin/stdout: position/diceroll/go -> bestmove)
'use strict';
const path = require('path');
const APP = path.resolve(__dirname, '..', '..', 'SnartSahkku');
const SS = require(path.join(APP, 'js', 'engine.js'));
require(path.join(APP, 'js', 'fen.js'));
const CPU = require(path.join(APP, 'js', 'cpu.js'));

const START_FEN = 'g-g-g-g-g-g-g-g-g-g-g-g-g-g-g/.-.-.-d-.-.-.-N-.-.-.-q-.-.-.' +
                  '/m-m-m-m-m-m-m-m-m-m-m-m-m-m-m w roll - 000 H2 0:0 0 *';
const DEC = { X: 'X', '2': 'II', '3': 'III', '_': '_' };
const UN = { X: 'X', II: '2', III: '3', _: '_' };

let LEVEL = 3;
let st = null;
const out = (s) => process.stdout.write(s + '\n');

function decodeTok(st_, tok) {
  for (const mv of SS.legalMoves(st_, st_.turn)) {
    const p = SS.piece(st_, mv.pid);
    let f, t;
    if (p.t === 's') {
      f = SS.soldierAt(p.o, p.at);
      t = SS.soldierAt(p.o, p.at + (mv.kind === 'rel' ? 1 : mv.v));
    } else if (mv.to !== undefined && mv.to !== null) {
      f = p.at; t = mv.to;
    } else {
      const r = SS.pathFor(st_, p, mv.dir, mv.v);
      if (!r.legal || !r.path.length) continue;
      f = p.at; t = r.path[r.path.length - 1];
    }
    if (SS.name(f) + '-' + SS.name(t) === tok) return mv;
  }
  return null;
}

function encodeTok(st_, mv) {
  const p = SS.piece(st_, mv.pid);
  if (p.t === 's') {
    const from = SS.soldierAt(p.o, p.at);
    const to = SS.soldierAt(p.o, p.at + (mv.kind === 'rel' ? 1 : mv.v));
    return SS.name(from) + '-' + SS.name(to);
  }
  if (mv.to !== undefined && mv.to !== null) return SS.name(p.at) + '-' + SS.name(mv.to);
  const r = SS.pathFor(st_, p, mv.dir, mv.v);
  if (!r.legal || !r.path.length) return null;
  return SS.name(p.at) + '-' + SS.name(r.path[r.path.length - 1]);
}

function applyToken(st_, tok, words, i) {
  if (tok === 'roll') {
    const faces = [words[i + 1], words[i + 2], words[i + 3]].map((f) => {
      if (!DEC[f]) throw new Error('ulovleg terning «' + f + '»');
      return DEC[f];
    });
    st_.dice = faces;
    st_.used = faces.map((d) => d === '_');
    st_.diceBy = st_.turn;
    st_.phase = 'move';
    return 4;
  }
  if (tok === 'pass') { SS.passRemaining(st_); SS.settle(st_); return 1; }
  if (tok === 'reroll') return 1;
  if (tok === 'resign') { st_.winner = 1 - st_.turn; st_.winReason = 'overgivelse'; st_.phase = 'over'; return 1; }
  if (/^[A-O][123]-[A-O][123]$/.test(tok)) {
    const mv = decodeTok(st_, tok);
    if (!mv) throw new Error('ulovleg trekk «' + tok + '»');
    SS.applyMove(st_, mv); SS.settle(st_);
    return 1;
  }
  throw new Error('ukjend token «' + tok + '»');
}

function parsePosition(words) {
  let i = 1, s2;
  if (words[i] === 'startpos') { s2 = SS.fromFEN(START_FEN); i++; }
  else if (words[i] === 'fen') {
    const rest = words.slice(i + 1);
    const mIx = rest.indexOf('moves');
    const fen = mIx >= 0 ? rest.slice(0, mIx).join(' ') : rest.join(' ');
    s2 = SS.fromFEN(fen);
    i = mIx >= 0 ? i + 1 + mIx + 1 : words.length;
  } else throw new Error('position: startpos|fen');
  if (words[i] === 'moves') i++;
  while (i < words.length) i += applyToken(s2, words[i], words, i);
  return s2;
}

function doGo() {
  if (!st || st.winner != null || st.phase === 'over') { out('bestmove none'); return; }
  if (st.phase !== 'move' || !st.dice || !st.dice.length) {
    out('bestmove none'); return;
  }
  if (!SS.legalMoves(st, st.turn).length) {
    out('bestmove ' + (SS.canRerollX(st) ? 'reroll' : 'pass')); return;
  }
  const mv = CPU.pick(st, LEVEL);
  out('bestmove ' + (mv ? encodeTok(st, mv) : 'none'));
}

let buf = '';
process.stdin.on('data', (d) => {
  buf += d.toString();
  let nl;
  while ((nl = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, nl).trim();
    buf = buf.slice(nl + 1);
    if (!line) continue;
    const words = line.split(/\s+/);
    const cmd = words[0];
    try {
      if (cmd === 'uci') { out('id name Classic CPU (FLOKK, level ' + LEVEL + ')'); out('uciok'); }
      else if (cmd === 'isready') out('readyok');
      else if (cmd === 'position') st = parsePosition(words);
      else if (cmd === 'diceroll') {
        const faces = words.slice(1, 4).map((f) => DEC[f] || f);
        st.dice = faces;
        st.used = faces.map((x) => x === '_');
        st.phase = 'move';
      }
      else if (cmd === 'go' || cmd === 'getmove') doGo();
      else if (cmd === 'setoption') {
        const i = words.indexOf('value');
        const v = i >= 0 ? parseInt(words[i + 1], 10) : NaN;
        if (!isNaN(v)) LEVEL = Math.max(1, Math.min(3, v));
      }
      else if (cmd === 'quit') process.exit(0);
    } catch (e) {
      out('info string error ' + e.message);
      out('bestmove none');
    }
  }
});
// arg: --level N
const ai = process.argv.indexOf('--level');
if (ai > 0) LEVEL = parseInt(process.argv[ai + 1], 10) || 3;
