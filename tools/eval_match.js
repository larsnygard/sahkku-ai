// eval_match — styrkemåling: to S-UCI-motorar mot kvarandre, DOM = ekte JS-referee.
//
// Dommaren er DEN EKAKTE SnartSahkku-js-regelen (vendor/engine.js + vendor/fen.js,
// kopiert frå ../SnartSahkku/js/), nøyaktig slik tests/suci.test.js driv ein motor:
//   turstart: position <replay m/ roll-token> → diceroll <3 fjes> → go
//   midt i tur: position <replay> → go            (dice+used kjem via roll-tokenet)
//   reroll:  position <replay> → diceroll <nye fjes> → go   (turen held fram)
//   pass:    turen sluttar; DOM-en snur sida sjølv
//
// To ting som må gjerast rett i DOM-en (og som braut den gamle versjonen):
//   1. SS.roll() set IKKJE st.phase = 'move' — det gjer GUI-en (js/main.js:120).
//      Utan det er SS.legalMoves() alltid tom → pool=[] → «ulovleg» kvart trekk.
//   2. Etter 'reroll' må replayen ha 'reroll' FØLGJENDES 'roll d1 d2 d3', og
//      GUI-en må sende diceroll på nytt med dei nye fjesa (S-UCI §5).
//
// Bruk:  node tools/eval_match.js "<kommando A>" "<kommando B>" [parti] [seed]
// T.d.   node tools/eval_match.js "node engines/sahkku-nn.js" \
//                                 "node ../SnartSahkku/engines/sahkku-ref.js" 20 7
//
// Sidene byter kvar part: A spelar Menn (w) i part 0,2,4… og Giera (b) i 1,3,5…
// (sáhkku er asymmetrisk, så ein rein A-som-Menn-benk ville vore skeiv).
'use strict';
const { spawn } = require('child_process');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const vroot = path.join(ROOT, 'vendor');
require(path.join(vroot, 'engine.js'));
require(path.join(vroot, 'fen.js'));
const SS = global.SS;

const START_FEN = 'g-g-g-g-g-g-g-g-g-g-g-g-g-g-g/.-.-.-d-.-.-.-N-.-.-.-q-.-.-.' +
                  '/m-m-m-m-m-m-m-m-m-m-m-m-m-m-m w roll - 000 H2 0:0 0 *';

const UNDEC = { X: 'X', II: '2', III: '3', _: '_' };      // interne fjes -> wire
const DEC = { X: 'X', '2': 'II', '3': 'III', '_': '_' };  // wire -> interne fjes
const diceStr = (faces) => faces.map((f) => UNDEC[f]).join(' ');
const sideName = (s) => (s === 0 ? 'Menn' : 'Giera');

// ---------------------------------------------------------------- motor-prosess
function spawnEng(cmdStr, label) {
  const parts = cmdStr.split(/\s+/).filter(Boolean);
  const p = spawn(parts[0], parts.slice(1), { stdio: ['pipe', 'pipe', 'pipe'] });
  p.stderr.on('data', () => {});
  const q = [];
  const waits = [];
  p.stdout.on('data', (d) => {
    d.toString().split('\n').filter((s) => s.length).forEach((line) => {
      if (waits.length) waits.shift()(line);
      else q.push(line);
    });
  });
  p.on('exit', (c) => { p._exit = c; });
  function readline(timeout) {
    return new Promise((res, rej) => {
      if (q.length) return res(q.shift());
      const t = setTimeout(() => rej(new Error(label + ': timeout (' + timeout + 'ms)')), timeout || 60000);
      waits.push((l) => { clearTimeout(t); res(l); });
    });
  }
  return {
    p, label, pid: p.pid,
    send: (s) => { p.stdin.write(s + '\n'); },
    read: (to) => readline(to),
    async bestmove(to) {
      for (;;) {
        const l = await readline(to || 60000);
        if (l.startsWith('bestmove')) return l.split(' ')[1];
        if (l.startsWith('error')) throw new Error(label + ': ' + l);
        // info … / movepool … : ignorer
      }
    },
    async handshake() {
      p.stdin.write('uci\n');
      for (;;) if ((await readline(30000)) === 'uciok') break;
      p.stdin.write('isready\n');
      for (;;) if ((await readline(30000)) === 'readyok') break;
      p.stdin.write('ucinewgame\n');
    },
    kill() { try { p.stdin.write('quit\n'); } catch (e) {} try { p.kill(); } catch (e) {} }
  };
}

// ------------------------------------------------------------- referee: dekod trekk
function decodeMove(st, tok) {
  for (const mv of SS.legalMoves(st, st.turn)) {
    const p = SS.piece(st, mv.pid);
    let f, t;
    if (p.t === 's') {
      f = SS.soldierAt(p.o, p.at);
      t = SS.soldierAt(p.o, p.at + (mv.kind === 'rel' ? 1 : mv.v));
    } else {
      const r = SS.pathFor(st, p, mv.dir, mv.v);
      if (!r.legal || !r.path.length) continue;
      f = p.at; t = r.path[r.path.length - 1];
    }
    if (SS.name(f) + '-' + SS.name(t) === tok) return mv;
  }
  return null;
}

function poolStrings(st) {
  const seen = new Set(), out = [];
  for (const mv of SS.legalMoves(st, st.turn)) {
    const p = SS.piece(st, mv.pid);
    let f, t;
    if (p.t === 's') {
      f = SS.soldierAt(p.o, p.at);
      t = SS.soldierAt(p.o, p.at + (mv.kind === 'rel' ? 1 : mv.v));
    } else {
      const r = SS.pathFor(st, p, mv.dir, mv.v);
      if (!r.legal || !r.path.length) continue;
      f = p.at; t = r.path[r.path.length - 1];
    }
    const s = SS.name(f) + '-' + SS.name(t);
    if (!seen.has(s)) { seen.add(s); out.push(s); }
  }
  return out;
}

// ------------------------------------------------------------------- eitt parti
// aSide: 0 = A spelar Menn, 1 = A spelar Giera
async function playGame(A, B, aSide, maxTurns) {
  const st = SS.fromFEN(START_FEN);
  SS.openingRoll(st, Math.random);
  const hist = [];
  const engines = { 0: aSide === 0 ? A : B, 1: aSide === 1 ? A : B };
  const replay = () => 'position startpos' + (hist.length ? ' moves ' + hist.join(' ') : '');
  let plies = 0;

  for (let turn = 0; turn < (maxTurns || 400) && st.winner == null; turn++) {
    // ---- turstart: DOM-en kastar terningane (motoren får aldri kaste sjølv) ----
    const faces = SS.roll(st, Math.random);
    if (!faces) throw new Error('roll() i fase «' + st.phase + '»: ' + SS.toFEN(st));
    st.phase = 'move';                 // <-- SS.roll gjer IKKJE dette; GUI-en gjer det
    hist.push('roll ' + diceStr(faces));
    let fresh = true;                  // nytt kast => diceroll skal ut

    for (;;) {
      const eng = engines[st.turn];
      eng.send(replay());
      if (fresh) { eng.send('diceroll ' + diceStr(st.dice)); fresh = false; }
      eng.send('go');
      const tok = await eng.bestmove();

      if (tok === 'none' || tok === 'resign') {
        st.winner = 1 - st.turn;
        st.winReason = 'motoren sa ' + tok;
        break;
      }
      if (tok === 'reroll') {
        if (!SS.canRerollX(st)) {
          throw new Error(eng.label + ': reroll utan X-omkast-rett FEN=' + SS.toFEN(st));
        }
        SS.rerollX(st, Math.random);
        st.phase = 'move';             // rerollX held turen, men set ikkje fasen
        hist.push('reroll');
        hist.push('roll ' + diceStr(st.dice));
        fresh = true;                  // spec §5: diceroll rett etter reroll
        plies++;
        continue;
      }
      if (tok === 'pass') {
        if (SS.legalMoves(st, st.turn).length || SS.canRerollX(st)) {
          throw new Error(eng.label + ': pass, men trekk/omkast fnst FEN=' + SS.toFEN(st));
        }
        hist.push('pass');
        SS.passRemaining(st);          // -> phase 'roll', turen snudd
        plies++;
        break;
      }
      const mv = decodeMove(st, tok);
      if (!mv) {
        throw new Error(eng.label + ': ulovleg «' + tok + '» FEN=' + SS.toFEN(st) +
                        ' pool=' + JSON.stringify(poolStrings(st)));
      }
      SS.applyMove(st, mv);
      hist.push(tok);
      plies++;
      if (st.winner != null) break;
      const s = SS.settle(st);
      if (s.over || s.turnEnded) break;
      // s.moves / s.reroll: same spelar held fram (reroll vert svara på neste runde)
    }
  }
  return {
    winner: st.winner, plies, reason: st.winReason || '',
    fen: SS.toFEN(st), histLen: hist.length
  };
}

// ----------------------------------------------------------------------- main
(async () => {
  const args = process.argv.slice(2);
  const [cmdA, cmdB, nStr, seedStr] = args;
  if (!cmdA || !cmdB) {
    console.log('bruk: node tools/eval_match.js "<cmd A>" "<cmd B>" [parti] [seed] [maks turar]');
    process.exit(1);
  }
  const n = parseInt(nStr || '20', 10);
  const seed = parseInt(seedStr || '7', 10);

  let a = seed >>> 0;                          // deterministisk Math.random
  Math.random = function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const A = spawnEng(cmdA, 'A'), B = spawnEng(cmdB, 'B');
  await A.handshake();
  await B.handshake();
  // Motorval (t.d. referansemotoren sitt «Level 1|2|3»):
  //   EVAL_SETOPT="Level=1"  /  EVAL_SETOPT_A=… EVAL_SETOPT_B=…
  const setopt = (e, s) => {
    const i = s.indexOf('=');
    e.send('setoption name ' + s.slice(0, i) + ' value ' + s.slice(i + 1));
  };
  const optAll = process.env.EVAL_SETOPT || '';
  if (optAll) { setopt(A, optAll); setopt(B, optAll); }
  if (process.env.EVAL_SETOPT_A) setopt(A, process.env.EVAL_SETOPT_A);
  if (process.env.EVAL_SETOPT_B) setopt(B, process.env.EVAL_SETOPT_B);

  let aw = 0, bw = 0, dr = 0, er = 0;
  const perSide = { 0: [0, 0, 0], 1: [0, 0, 0] };  // [A-vinst, B-vinst, uavgjort] når A er på sida
  const t0 = Date.now();
  for (let g = 0; g < n; g++) {
    const aSide = g % 2;               // alterner side for A
    let r = null, err = null;
    try { r = await playGame(A, B, aSide, 400); }
    catch (e) { err = e.message; }
    if (err) {
      er++;
      console.log('parti ' + (g + 1) + ' [A=' + sideName(aSide) + '] FEIL: ' + err);
      continue;
    }
    let who;
    if (r.winner === null || r.winner === undefined) {
      dr++; perSide[aSide][2]++; who = 'uavgjort (' + r.plies + ' halvtrekk, vakt)';
    } else {
      const aWon = r.winner === aSide;
      if (aWon) { aw++; perSide[aSide][0]++; } else { bw++; perSide[aSide][1]++; }
      who = (aWon ? 'A' : 'B') + ' (' + sideName(r.winner) + ') ' + r.plies + ' halvtrekk' +
            (r.reason ? ' {' + r.reason + '}' : '');
    }
    console.log('parti ' + (g + 1) + ' [A=' + sideName(aSide) + ']: ' + who);
  }
  const el = ((Date.now() - t0) / 1000).toFixed(0);
  console.log('SUM: A=' + aw + '  B=' + bw + '  uavgjort=' + dr + (er ? '  feil=' + er : '') +
              '  av ' + n + '  (' + el + ' s)');
  for (const s of [0, 1]) {
    const [x, y, d] = perSide[s];
    console.log('  A som ' + sideName(s) + ': ' + x + '-' + y + (d ? '-' + d + 'd' : ''));
  }
  const score = (aw + 0.5 * dr) / Math.max(1, n - er);
  console.log('  A-poengsum: ' + (100 * score).toFixed(1) + ' %  (seier=1, uavgjort=0,5)');
  A.kill(); B.kill();
  process.exit(er && er === n ? 2 : 0);
})().catch((e) => { console.error('eval_match: ' + (e && e.stack ? e.stack : e)); process.exit(2); });
