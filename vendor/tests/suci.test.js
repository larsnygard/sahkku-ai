// S-UCI 1.0-konsistensetest: køyrer ein motor (standard: engines/sahkku-ref.js,
// overstyr med SAHKKU_ENGINE="...") som ein ekte GUI: handshake, diceroll, go,
// dommar-regelkontroll av kvart bestmove, heile partiet til slutt.
'use strict';
const { spawn } = require('child_process');
const path = require('path');

const SS = require('../engine.js');
require('../fen.js');
const START_FEN = 'g-g-g-g-g-g-g-g-g-g-g-g-g-g-g/.-.-.-d-.-.-.-N-.-.-.-q-.-.-./m-m-m-m-m-m-m-m-m-m-m-m-m-m-m w roll - 000 H2 0:0 0 *';
const DEC = { X: 'X', '2': 'II', '3': 'III', '_': '_' };
const ENC = { X: 'X', II: '2', III: '3', _: '_' };
const rnd = (function () { let s = 0x1234abcd >>> 0; return function () { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; })();
function roll() { return [0, 1, 2].map(function () { return SS.DICE[Math.floor(rnd() * 4)]; }); }

let passed = 0, failed = 0;
function ok(cond, msg) { if (cond) passed++; else { failed++; console.log('FAIL: ' + msg); } }

const CMD = process.env.SAHKKU_ENGINE ? process.env.SAHKKU_ENGINE.split(' ')
  : [process.execPath, path.join(__dirname, '..', 'sahkku-ref.js')];
const eng = spawn(CMD[0], CMD.slice(1), { stdio: ['pipe', 'pipe', 'pipe'] });
let buf = '';
const lines = [];    // mottakne linjer enno ingen ventar på
const waiters = [];  // resolvarar frå readLine
eng.stdout.on('data', function (d) {
  buf += d.toString();
  let ix;
  while ((ix = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, ix).trim();
    buf = buf.slice(ix + 1);
    if (waiters.length) waiters.shift()(line);
    else lines.push(line);
  }
});
eng.stderr.on('data', function (d) { console.log('MOTOR-STDERR:', d.toString().slice(0, 300)); });

function send(line) { eng.stdin.write(line + '\n'); }
function readLine(timeoutMs) {
  if (lines.length) return Promise.resolve(lines.shift());
  return new Promise(function (resolve, reject) {
    waiters.push(resolve);
    setTimeout(function () { reject(new Error('motor svartar ikkje på ' + timeoutMs + ' ms')); }, timeoutMs || 15000);
  });
}
async function expect(pred, what, timeoutMs) {
  const line = await readLine(timeoutMs);
  if (!pred(line)) throw new Error('forventa ' + what + ', fekk: «' + line + '»');
  return line;
}

(async function () {
  // ---------- handshake ----------
  send('uci');
  let uciok = false, sawId = false, sawLevel = false;
  for (;;) {
    const l = await expect(function (x) { return true; }, 'handshake-linje');
    if (l.startsWith('id name')) sawId = true;
    if (l.startsWith('option name Level')) sawLevel = true;
    if (l === 'uciok') { uciok = true; break; }
    if (l.startsWith('id author')) ok(true, 'id author: ' + l);
  }
  ok(sawId, 'handshake: id name');
  ok(uciok, 'handshake: uciok');
  send('isready');
  await expect(function (l) { return l === 'readyok'; }, 'readyok');
  ok(true, 'isready -> readyok');
  send('ucinewgame');
  send('debug on');
  send('setoption name Level value 1');   // nivå 1: raskaste partiet

  // ---------- posisjonsmaskin ----------
  let st = SS.fromFEN(START_FEN);
  const hist = [];   // replay-token for partiet
  function diceStr(d) { return d.map(function (x) { return ENC[x]; }).join(' '); }
  function positionCmd() {
    // base = startpos + replay av trekk/pass/omkast (ALDRI roll-token:
    // terningane kjem utelukkande via diceroll, jf. spec §5)
    return 'position fen ' + START_FEN + (hist.length ? ' moves ' + hist.join(' ') : '');
  }

  // ---------- heile partiet som GUI ----------
  let rolls = 0, movesPlayed = 0, passes = 0, rerolls = 0, resigns = 0, infos = 0;
  let guard = 0;
  async function askGo(faces, turnStart) {
    // stillings-GUI: position (startpos + replay med roll-token) + go.
    // diceroll berre ved TURSTART (spec §2); midt i ein tur gjenopprettar
    // roll-tokenet i replayen terningar + used-masken nøyaktig.
    send(positionCmd());
    if (turnStart) send('diceroll ' + diceStr(faces));
    send('go');
    let bm = null;
    while (!bm) {
      const l = await expect(function (x) { return true; }, 'svar på go');
      if (l.startsWith('bestmove')) bm = l.slice(8).trim();
      else if (l.startsWith('info string position -> ') || l.startsWith('info string diceroll -> ')) {
        const engFen = l.slice(l.indexOf('-> ') + 3);
        const guiFen = SS.toFEN(st);
        if (engFen !== guiFen) {
          console.log('FEN-AVVIK (' + l.slice(0, 20) + '):');
          console.log('GUI : ' + guiFen);
          console.log('MOT : ' + engFen);
          console.log('hist: ' + hist.join(' | '));
          throw new Error('tilstandsavvik mellom GUI og motor');
        }
      }
      else if (l.startsWith('info')) infos++;
      else if (l.startsWith('error')) throw new Error('motoren feila under go: ' + l);
      else throw new Error('ulempa svarlinje: ' + l);
    }
    return bm;
  }
  function handleBm(bm) {
    // returnerer: 'move' | 'pass' | 'reroll' | 'resign'
    const pool = SS.legalMoves(st, st.turn);
    if (bm === 'none' || bm === 'resign') {
      resigns++;
      st.winner = 1 - st.turn; st.phase = 'over'; st.winReason = 'motor ga ' + bm;
      hist.push('resign');
      return 'resign';
    }
    if (bm === 'reroll') {
      ok(SS.canRerollX(st), 'reroll berre når X-omkast-regelen gjeld');
      hist.push('reroll');
      rerolls++;
      return 'reroll';
    }
    if (bm === 'pass') {
      ok(pool.length === 0 && !SS.canRerollX(st), 'pass berre når intet trekk finst');
      SS.passRemaining(st);
      hist.push('pass');
      passes++;
      return 'pass';
    }
    ok(/^\s*[A-O][123]-[A-O][123]\s*$/.test(bm), 'trekkformat: «' + bm + '»');
    const mv = findMove(st, bm);
    ok(mv !== null, 'bestmove «' + bm + '» er lovleg (pool: ' +
      pool.map(function (m) { return encMove(st, m); }).join(',') + ')');
    if (!mv) throw new Error('motoren spelte ulovleg trekk');
    hist.push(bm);
    SS.applyMove(st, mv);
    movesPlayed++;
    return 'move';
  }
  outer:
  while (st.winner == null && guard++ < 4000) {
    // éin tur: eitt kast, inntil 3 trekk (X>III>II), eventuelle X-omkast
    for (;;) {
      const faces = roll();
      rolls++;
      hist.push('roll ' + diceStr(faces));   // roll-token HØYRER replayen (spec §5)
      st.dice = faces.slice();
      st.used = faces.map(function (d) { return d === '_'; });
      st.diceBy = st.turn;
      st.phase = 'move';
      let turnOpen = true;
      let slotIx = 0;
      while (turnOpen && st.winner == null) {
        const bm = await askGo(faces, slotIx === 0);
        slotIx++;
        const kind = handleBm(bm);
        if (kind === 'resign') break outer;
        if (kind === 'reroll') break;               // nytt kast same spelar
        if (kind === 'pass') { turnOpen = false; break; }
        const s = SS.settle(st);
        if (s.over) break outer;
        if (s.turnEnded) { turnOpen = false; break; }
        if (s.reroll) {
          // dommar seier X-omkast er påkravd/tilbydd: la motoren få svare
          // (vi spelar GUI-rolla: kast sjølve nye X-terningar)
          hist.push('reroll');
          rerolls++;
          break;
        }
      }
      if (st.winner != null) break outer;
      if (st.phase === 'roll' || SS.settle(st).turnEnded) break;  // neste spelar
    }
  }
  ok(st.winner != null, 'partiet gjekk til slutt (vakt ' + guard + ')');
  ok(movesPlayed >= 5, 'minst 5 trekk spelte (' + movesPlayed + ')');
  ok(rolls >= 3, 'minst 3 terningskast (' + rolls + ')');

  // ---------- avslutting + nyr start ----------
  send('ucinewgame');
  send('isready');
  await expect(function (l) { return l === 'readyok'; }, 'readyok etter ucinewgame');
  ok(true, 'ucinewgame nullar motorstaten cleanly');

  // ---------- movelist + position startpos ----------
  send('position startpos');
  send('diceroll X X _');
  send('movelist');
  let poolLines = 0;
  for (;;) {
    const l = await expect(function (x) { return true; }, 'movelist-linje');
    if (l === 'movepool done') break;
    if (l.startsWith('info string')) continue;   // debug-logg (debug on) tolererast
    ok(l.startsWith('info movepool '), 'movelist-format: ' + l);
    poolLines++;
  }
  ok(poolLines >= 1, 'movelist gav minst eitt trekk (' + poolLines + ')');

  // ---------- FEN-runde: toFEN -> fromFEN -> identical ----------
  const f1 = SS.toFEN(st);
  const st2 = SS.fromFEN(f1);
  ok(SS.toFEN(st2) === f1, 'FEN roundtrip stabil: ' + f1);

  // ---------- feilhandtering ----------
  send('position fen ugjenkjennelig fen her');
  await expect(function (l) { return l.startsWith('error'); }, 'error på øydelagd FEN');
  ok(true, 'øydelagd FEN -> error, motoren lever');
  send('isready');
  await expect(function (l) { return l === 'readyok'; }, 'readyok etter feil');
  ok(true, 'motor overlever feil (Level-A-krav)');

  send('quit');
  await new Promise(function (r) { eng.on('exit', r); setTimeout(r, 3000); });
  console.log('kast: ' + rolls + ' · trekk: ' + movesPlayed + ' · pass: ' + passes +
    ' · omkast: ' + rerolls + ' · resign: ' + resigns + ' · info-linjer: ' + infos);
  console.log(passed + ' pass, ' + failed + ' fail');
  process.exit(failed ? 1 : 0);
})().catch(function (e) {
  console.log('S-UCI TEST FEILA: ' + e.message);
  eng.kill('SIGKILL');
  process.exit(1);
});

// trekk-koding (same som motoren): {pid,dir,v,kind} -> "A1-B1"
function encMove(st, mv) {
  const p = SS.piece(st, mv.pid);
  if (p.t === 's') {
    const from = SS.soldierAt(p.o, p.at);
    const to = SS.soldierAt(p.o, p.at + (mv.kind === 'rel' ? 1 : mv.v));
    return SS.name(from) + '-' + SS.name(to);
  }
  return SS.name(p.at) + '-' + SS.name(SS.moveDest(st, mv));
}
function findMove(st, str) {
  const lms = SS.legalMoves(st, st.turn);
  for (const mv of lms) if (encMove(st, mv) === str) return mv;
  return null;
}
