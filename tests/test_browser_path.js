// Nettlesar-stien: last engines/sahkku-nn.js i ein VM-kontekst UTAN Node-globalar
// (require/module/process finst ikkje i ein nettlesar), last vektene med
// loadWeights() slik ein nettlesar ville gjort med fetch + JSON.parse, og
// samanlikn netForward med Node-stien (features-test) for same stilling.
//
// Bruk: node tests/test_browser_path.js [sti-til-sahkku-nn.js]
'use strict';
const vm = require('vm');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const ROOT = path.join(__dirname, '..');
const ENGINE = process.argv[2] || path.join(ROOT, 'engines', 'sahkku-nn.js');

const src = fs.readFileSync(ENGINE, 'utf8');
const ctx = {
  console, Math, JSON, Object, Array, Number, isNaN, Uint8Array, Float32Array,
  TextDecoder,
  atob: (b) => Buffer.from(b, 'base64').toString('binary'),
};
vm.createContext(ctx);
vm.runInContext(src, ctx);
if (!ctx.NN || !ctx.NN.exported) {
  console.log('FAIL: NN.exported finst ikkje når require/module manglar');
  process.exit(1);
}
const weightsPath = ENGINE.replace(/\.js$/, '.weights.json');
ctx.NN.exported.loadWeights(JSON.parse(fs.readFileSync(weightsPath, 'utf8')));

// same stilling gjennom begge stiane
const FEN = 'g-g-g-g-g-g-g-g-g-g-g-g-g-g-g/.-.-.-d-.-.-.-N-.-.-.-q-.-.-.' +
            '/m-m-m-m-m-m-m-m-m-m-m-m-m-m-m w move X2_ 000 H2 0:0 0 *';
const N = 6;                     // fleire stillingar: starten + tilfeldig spel
const st0 = ctx.NN.exported.SS.fromFEN(FEN);
const enc = ctx.NN.exported.encodeState(st0);
const ids = ctx.NN.exported.legalIds(st0);
const r = ctx.NN.exported.netForward(enc.planes, enc.glob);

const nodeOut = JSON.parse(execFileSync(process.execPath, [ENGINE, 'features-test'], {
  input: JSON.stringify([{ fen: FEN }]),
  maxBuffer: 64 << 20,
}))[0];

let fails = 0;
function check(cond, msg) { if (!cond) { console.log('FAIL: ' + msg); fails++; } }
check(ids.length > 0, 'ingen lovlege aksjonar i VM-stien');
check(JSON.stringify(ids.map((x) => x[0])) === JSON.stringify(nodeOut.legal),
      'legal-pool ulik: vm ' + ids.map((x) => x[0]) + ' vs node ' + nodeOut.legal);
let dP = 0, dV = Math.abs(r.value - nodeOut.value);
ids.forEach((x, i) => { dP = Math.max(dP, Math.abs(r.policy[x[0]] - nodeOut.policy[i])); });
check(dP < 1e-3, 'policy avvik ' + dP);
check(dV < 1e-3, 'value avvik ' + dV);
if (fails) process.exit(1);
console.log('NETTLESAR-STI OK: modulen lastar utan Node-globalar, ' + ids.length +
            ' aksjonar, Δpolicy ' + dP.toExponential(1) + ', Δvalue ' + dV.toExponential(1));
