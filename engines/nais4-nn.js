/* sahkku-nn — S-UCI 1.0 neural motor for sáhkku (spec: docs/SAHKKU-UCI.md).
 * Reglane er DEN EKAKTE JS-motoren: js/engine.js + js/fen.js frå SnartSahkku
 * blir INJISERTE under ved `python -m sahkku_ai.export` (§RULES§/§FEN§-markørar)
 * — ingen avskriven kopi, slik at reglane ALLTID er 1:1 med DOM-en.
 * Nettopp: nais4-nn.weights.json (JSON, base64 float32), køyrd i rein JS her —
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
// SnartSáhkku — Kjerne: Lágesvuotna-sáhkku (ren logikk, ingen DOM)
//
// Brett: 3 rekker (ráđat) x 15 linjer (sárgát). Indes: rad0(nede)=0..14,
// rad1(midten)=15..29, rad2(øverst)=30..44.
// Soldatenes ferd er en 75-punkters syklus (tallsyklusen):
//   P1: rad0 A->O, rad1 O->A, rad2 A->O, rad1 O->A  ...
//   P2: rad2 O->A, rad1 A->O, rad0 O->A, rad1 A->O  ...
// Kongen/dronninga går i *rette linjer*: langs tallsyklusen (begge veier)
// eller loddrett (opp/ned, samme kolonne), uten å snu underveis.
// Slottet H2=22. X-merker L2=18, D2=26.
(function (global) {
  'use strict';

  var N = 45, W = 15;
  var DICE = ['X', 'II', 'III', '_'];
  var RANK = { X: 0, III: 1, II: 2, '_': 3 };
  var CASTLE = 22;

  function pt(i) {
    if (i < 15) return { col: i, row: 0 };
    if (i < 30) return { col: 29 - i, row: 1 };
    return { col: i - 30, row: 2 };
  }
  function idx(col, row) { return row === 0 ? col : row === 1 ? 29 - col : 30 + col; }
  function name(i) { var p = pt(i); return String.fromCharCode(65 + p.col) + (p.row + 1); }
  function rowLine(row) { var o = []; for (var c = 0; c < 15; c++) o.push(idx(c, row)); return o; }
  // felt som «A1»..«O3» (kolonne A–O, rekkje 1–3; rekkje 2 speilt frå P1-venstre)
  function sqName(i, flip) {
    var c, r;
    if (i < 15) { c = i; r = 1; }
    else if (i < 30) { c = 29 - i; r = 2; }
    else { c = i - 30; r = 3; }
    if (flip) c = 14 - c;
    return String.fromCharCode(65 + c) + r;
  }
  function mod(i, m) { i = i % m; return i < 0 ? i + m : i; }

  // tallsyklusen: t(0..74) -> brettindeks, per spiller
  var TOUR1 = (function () {
    var t = [], i;
    for (i = 0; i < 15; i++) t.push(idx(i, 0));        // rad0 A->O
    for (i = 0; i < 15; i++) t.push(idx(14 - i, 1));   // rad1 O->A
    for (i = 0; i < 15; i++) t.push(idx(i, 2));        // rad2 A->O
    for (i = 0; i < 15; i++) t.push(idx(14 - i, 1));   // rad1 O->A
    return t;                                          // lengde 75
  })();
  function tour2Of(t) { var i = pt(TOUR1[t]); return idx(14 - i.col, 2 - i.row); }
  function soldierAt(pl, t) { return pl === 0 ? TOUR1[mod(t, 60)] : tour2Of(mod(t, 60)); }
  // startposisjoner: P1-soldat på kol c (A1..O1) står på t=c (tallsyklus 0..14
  // = A1..O1); P2-soldat på kol c (O3..A3) står på t=14-c (syklusens speiling:
  // t=0 => O3, t=14 => A3). «Framst i rekka» = høyest t.
  function startT(pl, col) { return pl === 0 ? col : 14 - col; }

  // ---------- konge/dronning: rette bevegninger i fire retninger ----------
  // Dokumentet: kongen går «til høyre og venstre, og opp og ned», «ikke snu
  // underveis» — et trekk med verdi v = v steg i SAMME retning. Retningene
  // (sett fra P1, A1 nede til venstre):
  //   R: col+1;  O1->O2 og O2->O3 (hjørnene fortsetter «rett fram»)
  //   L: col-1;  A2->A3 (dokumentets eksempel: B2 -II-> A3)
  //   F: rad0->rad1->rad2, samme kolonne (stopper på rad 2)
  //   B: rad2->rad1->rad0, samme kolonne (stopper på rad 0)
  // Alle stegene i ett kast må ha samme retning (dir = 'R','L','F','B').
  var DIRS = ['R', 'L', 'F', 'B'];
  function stepDir(i, dir) {
    var p = pt(i), c = p.col, r = p.row;
    if (dir === 'R') {
      if (r === 0) return c < 14 ? idx(c + 1, 0) : idx(14, 1);   // O1 -> O2
      if (r === 1) return c < 14 ? idx(c + 1, 1) : idx(14, 2);   // O2 -> O3
      return c < 14 ? idx(c + 1, 2) : null;                       // stopper ved O3
    }
    if (dir === 'L') {
      if (r === 1) return c > 0 ? idx(c - 1, 1) : idx(0, 2);      // A2 -> A3 (dok. B2-II->A3)
      return c > 0 ? idx(c - 1, r) : null;                        // A1/A3: stopper
    }
    if (dir === 'F') {
      if (r === 0) return idx(c, 1);                              // opp: A1 -> A2, O1 -> O2
      if (r === 1) return idx(c, 2);                              // opp: B2 -> B3 (dok: kongen kan «rett opp»), A2 -> A3 (loddrett)
      return null;
    }
    if (dir === 'B') {
      if (r === 2) return idx(c, 1);                               // opp: A3->A2 … O3->O2 (fremst høgre hjørnet: nedstigen på ruta)
      if (r === 1) return idx(c, 0);
      return null;
    }
    return null;
  }
  // banen for v steg i dir; null hvis veggen stopper før v steg
  function pathFor(st, p, dir, v) {
    var path = [], cur = p.at, i, q;
    for (i = 0; i < v; i++) {
      q = stepDir(cur, dir);
      if (q == null) return { path: [], legal: false };
      path.push(q); cur = q;
    }
    return { path: path, legal: true };
  }
  // retninger tilgjengelig fra felt i med verdi v (vegg sjekkes her)
  function freeDirsAt(i, v) {
    var out = [];
    for (var d = 0; d < DIRS.length; d++) {
      var cur = i, okDir = true;
      for (var k = 0; k < v; k++) {
        var q = stepDir(cur, DIRS[d]);
        if (q == null) { okDir = false; break; }
        cur = q;
      }
      if (okDir) out.push(DIRS[d]);
    }
    return out;
  }
  // landingsbetingelser (kanLand): sant om p kan lande på land.
  // Dokumentet: hjemmestående brikker er passive — landing på linje med
  // uutløste brikker er forbudt (soldierLandingOk). PASSENDE brikker på
  // banen er fri: «linjene ... kan du overhodet ikke flytte INN PÅ» —
  // bare landing er regulated, gjennomfart fri.
  function canLand(st, p, land) {
    var on = occ(st, land), i, q;
    // KONGEN FYRST, UAVHENGIG AV REKKJEFØLGJA I occ (Lars 2026-10-28):
    // kapslar ein kongen, står kongen og soldaten saman på feltet — og det er
    // den einaste gongen. Difor kan ingen egne brikker (soldat eller dronning)
    // lande på EIGEN konge. Utan denne førehandssjekken kunne eit tredje
    // stykke kome inn berre fordi ein eigen soldat låg fyrst i occ og løkka
    // under returnerte true før ho nådde kongen.
    if (p.t !== 'k') {
      for (i = 0; i < on.length; i++) {
        q = on[i];
        if (q !== p && q.t === 'k' && q.o != null && q.o === p.o) return false;
      }
    }
    for (i = 0; i < on.length; i++) {
      q = on[i];
      if (q === p) continue;
      if (p.t === 's') {
        if (q.t === 'k') {
          if (q.o === p.o) return false;
          if (st.nesseby && q.o == null) return false;   // Nesseby: ikkje aktivert enno
          continue;
        }
        if (q.t === 'q' && q.o !== p.o) continue;      // slå fiendtlig dronning
        if (q.t === 'q' && q.o === p.o) return false;  // egen dronning: nektet
        // soldater: egne tillatt (stable), fiender slås, hjemmebrikker:
        if (isHome(q) && q.o === p.o) return true;     // egen hjemrad: se soldatLandingOk
        if (isHome(q)) return false;                   // inn på fiendes hjemrad: nektet
        return true;
      }
      if (p.t === 'k') {
        // Kongen KAPRAR som andre brikker (Lars 2026-10-26): han kan lande på
        // og slå ut fiendtlege soldatar og dronninga (og dermed vinne), men
        // ikkje på eigne brikker eller på uutløyste (heimeståande) brikker.
        if (q.t === 'k') return false;          // berre éin konge — han kan ikkje lande på seg sjølv
        if (q.o === p.o) return false;          // eigne brikker: nei
        if (isHome(q)) return false;            // uutløyst brikke på heimrad: nei
        return true;                            // fiende: kaprast
      }
      if (p.t === 'q') {
        if (q.t === 's' && q.o === p.o) return false;  // deler ikke felt med egne
        if (q.t === 'q' && q.o === p.o) return false;
        if (q.t === 'k') {
          if (st.nesseby && q.o == null) return false;  // Nesseby: ikkje aktivert enno
          continue;                                     // kapre kongen
        }
        if (isHome(q)) return false;                   // hjemmestående: kan ikke flyttes inn på
        return true;                                   // slår fiendesoldat / fiendedronning
      }
    }
    return true;
  }
  // soldat kan aldri lande der NOEN(uutløst) står (hjemraden er blokkert
  // for alle — dokumentasjonen: «linjene der det står brikker som ikke er
  // løsgjort kan du overhodet ikke flytte inn på»)
  function soldierLandingOk(st, p, land) {
    var on = occ(st, land), i;
    for (i = 0; i < on.length; i++) if (isHome(on[i])) return false;
    if (p.t === 's' && p.o === 0 && land === 0 && p.released === false) {}
    return true;
  }

  // ---------- tilstand ----------
  // Nesseby-regelen (valfri): kongen er IKKJE kaprbar ved landing før nokon
  // har nådd motstandarens heimrad. Den fyrste soldaten som når dit kaprar
  // kongen for si side; etter det gjeld vanlege landingskapringar.
  // (Konge utan eigar = ikkje aktivert, so me treng ingen eigen «aktivert»-flagg.)
  function newGame(opts) {
    opts = opts || {};
    var pieces = [], id = 0, pl, col;
    for (pl = 0; pl < 2; pl++) {
      for (col = 0; col < 15; col++) {
        pieces.push({
          id: id++, t: 's', o: pl, seq: col,
          at: startT(pl, col), t0: startT(pl, col), released: false, captured: false
        });
      }
      pieces.push({
        id: id++, t: 'q', o: pl, seq: -1,
        at: pl === 0 ? 18 : 26, t0: pl === 0 ? 18 : 26, released: false, captured: false
      });
    }
    pieces.push({ id: id++, t: 'k', o: null, seq: -1, at: CASTLE, t0: CASTLE, released: true, captured: false });
    return {
      v: 1, pieces: pieces, turn: opts.first != null ? opts.first : 0,
      phase: 'roll', dice: [], used: [], diceBy: null, log: [],
      winner: null, winReason: '', lastMove: null,
      kings: 0, kills: [0, 0], moves: 0,
      queenMode: opts.queenMode !== false,
      nesseby: opts.nesseby === true        // valfri regel, sjå newGame()
    };
  }
  // soldaters plassering: lagre tallsyklus-posisjon i .at for soldater,
  // brettindeks for konge/dronning. Hjelpefunksjoner:
  function satAt(st, p, land) { p.at = land; }
  function occBrett(st, i) {
    var out = [], k, p, a;
    for (k = 0; k < st.pieces.length; k++) {
      p = st.pieces[k];
      if (p.captured) continue;
      a = p.t === 's' ? soldierAt(p.o, p.at) : p.at;
      if (a === i) out.push(p);
    }
    return out;
  }
  function occ(st, i) { return occBrett(st, i); }
  function piece(st, pid) {
    for (var k = 0; k < st.pieces.length; k++) if (st.pieces[k].id === pid) return st.pieces[k];
    return null;
  }
  function soldiers(st, o) {
    return st.pieces.filter(function (p) { return p.t === 's' && p.o === o; });
  }
  function queen(st, o) {
    for (var k = 0; k < st.pieces.length; k++) {
      var p = st.pieces[k];
      if (p.t === 'q' && p.o === o && !p.captured) return p;
    }
    return null;
  }
  function king(st) {
    for (var k = 0; k < st.pieces.length; k++) if (st.pieces[k].t === 'k') return st.pieces[k];
    return null;
  }
  function isHome(p) { return (p.t === 's' || p.t === 'q') && !p.released && !p.captured; }
  function atIdx(st, p) { return p.t === 's' ? soldierAt(p.o, p.at) : p.at; }

  // Kvar landar eit trekk? Soldat: +v i syklusen (eller +1 ved løysing).
  // Konge/dronning: trekket ber landingsfeltet i `to` (tallsyklusen snor seg);
  // dronninga si X-løysing er ei rett line og vert rekna med pathFor.
  function moveDest(st, mv) {
    var p = piece(st, mv.pid);
    if (p.t === 's') return soldierAt(p.o, p.at + (mv.kind === 'rel' ? 1 : mv.v));
    if (mv.to !== undefined && mv.to !== null) return mv.to;
    var r = pathFor(st, p, mv.dir, mv.v);
    return r.path[r.path.length - 1];
  }

  // ---------- terninger ----------
  function rollDice(rnd) {
    var d = [], i;
    for (i = 0; i < 3; i++) d.push(DICE[Math.floor((rnd || Math.random)() * 4)]);
    return d;
  }
  function roll(st, rnd) {
    if (st.phase !== 'roll') return null;
    st.dice = rollDice(rnd);
    st.used = st.dice.map(function (d) { return d === '_'; }); // blank = ingenting å gjøre
    st.diceBy = st.turn;
    logPly(st).roll = (logPly(st).roll || []).concat([st.dice.slice()]);
    return st.dice.slice();
  }
  // ---------- spellog (for notasjon + gjenopptak) ----------
  // Eitt «ply» per kast: {t: side, roll: [terningkast...], mv: [hendings-objekt...]}
  function logPly(st) {
    if (!st.log) st.log = [];
    var last = st.log[st.log.length - 1];
    if (last && last.t === st.turn && last.closed !== true) return last;
    var ply = { t: st.turn, roll: [], mv: [], closed: false };
    st.log.push(ply);
    return ply;
  }
  function closePly(st) {
    var last = st.log && st.log[st.log.length - 1];
    if (last) last.closed = true;
  }
  function logReroll(st) {
    var ply = logPly(st);
    ply.roll.push(st.dice.slice());
    ply.rerolls = (ply.rerolls || 0) + 1;
  }
  function openingRoll(st, rnd) {
    var ev = [];
    for (;;) {
      var a = rollDice(rnd), b = rollDice(rnd);
      var xa = a.filter(function (d) { return d === 'X'; }).length;
      var xb = b.filter(function (d) { return d === 'X'; }).length;
      ev.push({ p0: a, p1: b });
      if (xa !== xb) {
        st.log = [{ t: 'open', rolls: ev, first: xa > xb ? 0 : 1 }];
        return { rolls: ev, first: xa > xb ? 0 : 1 };
      }
    }
  }
  function activeRank(st) {
    // Blank ('_') er eit dødt fjes: ho gjev ingen trekk og skal aldri verte
    // vald som aktiv terning (diceLeft hoppar òg over henne). Utan dette
    // kunne eit ubrukt blankt fjes verte lese som «II» (v=2) — t.d. rett
    // etter eit omkast som gav blank.
    var best = 4, i;
    for (i = 0; i < 3; i++) {
      if (st.used[i] || st.dice[i] === '_') continue;
      if (RANK[st.dice[i]] < best) best = RANK[st.dice[i]];
    }
    return best === 4 ? null : best;
  }
  function activeValue(st) {
    var rk = activeRank(st);
    if (rk == null) return 0;
    return rk === RANK.X ? 1 : rk === RANK.III ? 3 : 2;
  }
  function diceLeft(st) {
    var n = 0;
    for (var i = 0; i < 3; i++) if (!st.used[i] && st.dice[i] !== '_') n++;
    return n;
  }
  // Kvar X-terning er si EIGA handling (Lars 2026-10-27): «Xs can be
  // rerolled individually». Slår ein to X, kan ein halde på den eine og
  // kaste om den andre. Utan `slot` kastar vi om den fyrste ubrukte X-en —
  // det er kva protokollen (S-UCI `reroll`) og CPU-en gjer.
  function rerollX(st, rnd, slot) {
    if (!canRerollX(st)) return null;
    var i = (slot === undefined || slot === null) ? -1 : slot | 0;
    if (i >= 0) {
      if (st.dice[i] !== 'X' || st.used[i]) return null;   // berre ein ubrukt X
    } else {
      for (var k = 0; k < 3; k++) if (st.dice[k] === 'X' && !st.used[k]) { i = k; break; }
      if (i < 0) return null;
    }
    var nf = rollDice(rnd)[0];
    st.dice[i] = nf;
    st.used[i] = (nf === '_');     // eit nytt blankt fjes er brukt med ein gong
    logReroll(st);
    return st.dice.slice();
  }
  // X-OMKAST — sahkku.no-reglen, ordrett: «kan du velge om du vil bruke
  // X-ene eller kaste terningene med X om igjen inntil du får noe anna,
  // GITT AT DU IKKE KAN FLYTTE NOE ANNA». Dvs. frivillig valg, men
  // avgrensa til tilfelle der X (og resten av kastet) ikkje kan brukast.
  // (Lokale variantar — Wikipedia/Berg-Nordlie — har fritt omkast; vi
  // følgjer dokumentet.) II/III/blank er faste og kan ikkje kastast om.
  function canRerollX(st) {
    // Lars 2026-10-26: «Du kan alltid kaste om X, unntatt hvis aktivering av
    // en brikke er det eneste du kan gjøre.» Ein spelar som aldri aktiverer
    // noko ville elles kunne blokkere spelet for alltid — difor er omkast
    // forbode berre når sida ikkje har éi einaste aktiv (utløyst) brikke.
    if (st.winner != null || st.phase !== 'move') return false;
    if (activeRank(st) !== RANK.X) return false;
    var me = st.turn, i, p, hasActive = false;
    for (i = 0; i < st.pieces.length; i++) {
      p = st.pieces[i];
      if (p.captured) continue;
      if (p.t === 'k' ? p.o === me : (p.o === me && p.released)) { hasActive = true; break; }
    }
    if (hasActive) return true;                       // alltid lov
    return legalMoves(st, me).length === 0;           // utan aktive: berre om inga vekking finst
  }

  // konge/dronning: alle lovlige trekk med verdi v: v steg i SAMME
  // retning (ingen snuing underveis — dokumentets regel).
  // Fritt pass gjennom alt (soldater, hjemmestående, kongen) — landing er
  // det som reguleres av canLand.
  // ---- tallsyklusen (åttetalet): konge og dronning går LANGS ho ----
  // Brikka går v steg framover eller bakover i sin eigen side sin syklus, kan
  // hoppe over brikker (berre landingsfeltet er regulert) og må lande på eit
  // felt ho kan lande på (canLand). Fyrste-syklus-konvensjonen: ei brikke på
  // midtrekka tel som fyrste pass — same felt ligg på to syklusposisjonar.
  var CYC = [];
  function cycleIndexOf(pl, sq) {
    var m = CYC[pl];
    if (!m) { m = CYC[pl] = {}; for (var t = 0; t < 60; t++) { var s = soldierAt(pl, t); if (m[s] === undefined) m[s] = t; } }
    return m[sq] === undefined ? -1 : m[sq];
  }
  // Loddrett: konge og dronning går rett OPP eller rett NED i same kolonne
  // (Lars 2026-10-27: «kongen flyttar seg som dronninga, men opp/ned var
  // blokkert»). stepDir('F'/'B') fanst alt, men vart aldri kalla for konge
  // og dronning — difor var berre syklusvegen (åttetalet) tilgjengeleg.
  // Banen kan passere brikker; berre landinga er regulert (canLand).
  function genVerticalMoves(st, p, v) {
    var res = [], dirs = ['F', 'B'], k, r2, land;
    for (k = 0; k < dirs.length; k++) {
      r2 = pathFor(st, p, dirs[k], v);
      if (!r2.legal) continue;
      land = r2.path[r2.path.length - 1];
      if (!canLand(st, p, land)) continue;
      res.push({ pid: p.id, dir: dirs[k], v: v, kind: 'mv' });
    }
    return res;
  }

  function genCycleMoves(st, p, v) {
    var res = [], sq = p.at, t0 = cycleIndexOf(p.o, sq), s, land;
    if (t0 < 0) return res;
    for (s = -1; s <= 1; s += 2) {
      land = soldierAt(p.o, t0 + s * v);
      if (land === sq) continue;              // heil runde attende til same felt
      if (!canLand(st, p, land)) continue;
      res.push({ pid: p.id, dir: s > 0 ? 'F' : 'B', v: v, kind: 'mv', to: land, cyc: s });
    }
    return res;
  }
  // ---------- lovlige trekk ----------
  // Trekk: {pid, dir, v, kind}. Soldater: dir +1 (fremover), v = verdi.
  // Løsning: kind 'rel' v=1.
  function legalMoves(st, me) {
    if (st.phase !== 'move' || st.winner != null) return [];
    var out = [], rk = activeRank(st), v = activeValue(st);
    if (rk == null) return out;
    var i, p, land, res, l;
    if (rk === RANK.X) {
      // løsne soldat: den fremste i rekka (høyest syklusposisjon)
      var home = soldiers(st, me).filter(function (s) { return !s.released && !s.captured; });
      if (home.length) {
        var front = null;
        for (i = 0; i < home.length; i++) if (!front || home[i].at > front.at) front = home[i];
        land = soldierAt(me, front.at + 1);
        if (soldierLandingOk(st, front, land) && canLand(st, front, land))
          out.push({ pid: front.id, dir: +1, v: 1, kind: 'rel' });
      }
      // løsne dronninga: 1 steg fritt i en av de fire retningene
      var q = queen(st, me);
      if (q && !q.released && st.queenMode) {
        var dirsQ = DIRS;
        for (i = 0; i < dirsQ.length; i++) {
          res = pathFor(st, q, dirsQ[i], 1);
          if (res.legal && soldierLandingOk(st, q, res.path[0]) && canLand(st, q, res.path[0]))
            out.push({ pid: q.id, dir: dirsQ[i], v: 1, kind: 'rel' });
        }
      }
    }
    // X er eit 1-TAL (Lars 2026-10-26): ho kan vekkje ei inaktiv brikke
    // (soldat/dronning — flyttet er ein del av vekkinga) OG flytte ei
    // utløyst brikke eitt steg: soldat framover i tallsyklusen, dronning og
    // konge eitt steg langs syklusen. Vekkinga er alt lagd til ovanfor; den
    // generelle løkka nedanfor gjev 1-stegstrekka (v = activeValue = 1).
    for (i = 0; i < st.pieces.length; i++) {
      p = st.pieces[i];
      if (p.captured) continue;
      if (p.t === 's') {
        if (p.o !== me || !p.released) continue;
        // Tallsyklusen: kun FRAMOVER v steg (dir +1). Banen er FRI:
        // man kan PASSE alle brikker (soldater, konge, dronning — også
        // dronninga når hun står hjemmestående); det er bare LANDING
        // som er regulert (canLand + soldierLandingOk: inn på linje med
        // uutløste brikker kan man «overhodet ikke flytte inn på»).
        var destS = soldierAt(me, p.at + v);
        if (!soldierLandingOk(st, p, destS) || !canLand(st, p, destS)) continue;
        out.push({ pid: p.id, dir: +1, v: v, kind: 'mv' });
      } else if (p.t === 'q') {
        if (!st.queenMode || p.o !== me || !p.released) continue;
        var qm = genCycleMoves(st, p, v);
        for (var z = 0; z < qm.length; z++) out.push(qm[z]);
        var qv = genVerticalMoves(st, p, v);
        for (var zv = 0; zv < qv.length; zv++) out.push(qv[zv]);
      } else if (p.t === 'k') {
        if (p.o !== me) continue;
        var km2 = genCycleMoves(st, p, v);
        for (var z2 = 0; z2 < km2.length; z2++) out.push(km2[z2]);
        var kv = genVerticalMoves(st, p, v);
        for (var zk = 0; zk < kv.length; zk++) out.push(kv[zk]);
      }
    }
    return out;
  }

  // ---------- utfør ----------
  function useDie(st, v) {
    var face = v === 1 ? 'X' : v === 2 ? 'II' : 'III';
    for (var i = 0; i < 3; i++) if (!st.used[i] && st.dice[i] === face) { st.used[i] = true; return i; }
    return -1;
  }
  function applyMove(st, mv) {
    var ev = [], p = piece(st, mv.pid), me = st.turn;
    useDie(st, mv.v);
    var fromIdx, land;
    if (p.t === 's') {
      fromIdx = soldierAt(p.o, p.at);
      land = mv.kind === 'rel' ? soldierAt(p.o, p.at + 1) : soldierAt(p.o, p.at + mv.v);
      p.at = mv.kind === 'rel' ? p.at + 1 : p.at + mv.v;
    } else {
      fromIdx = p.at;
      if (mv.to !== undefined && mv.to !== null) {
        // konge/dronning langs tallsyklusen: landingsfeltet er rekna ut på
        // førehand (syklusen snor seg, so dir/v åleine gjev ikkje feltet)
        land = mv.to;
      } else {
        var res = pathFor(st, p, mv.dir, mv.v);   // dronninga sin X-løysing
        land = res.path[res.path.length - 1];
      }
      p.at = land;
    }
    if (mv.kind === 'rel') p.released = true;
    ev.push({ type: 'mv', pid: p.id, from: fromIdx, to: land, by: me, v: mv.v, dir: mv.dir, rel: mv.kind === 'rel' });
    var lg = { pid: p.id, t: p.t, o: p.o, from: fromIdx, to: land, v: mv.v, dir: mv.dir, rel: mv.kind === 'rel', k: 0, kg: false };
    logPly(st).mv.push(lg);
    st.lastMove = { pid: p.id, from: fromIdx, to: land, by: me };
    st.moves++;
    // landingseffekter
    var on = occ(st, land), i, q;
    for (i = 0; i < on.length; i++) {
      q = on[i];
      if (q === p) continue;
      if (q.t === 'k') {
        q.o = me; st.kings++;
        lg.kg = true;
        ev.push({ type: 'king', by: me, at: land });
        continue;
      }
      if (q.o === me) continue;                       // egne soldater: stable
      if (q.t === 's') {
        q.captured = true; q.released = true;
        st.kills[me]++;
        lg.k++; lg.cap = (lg.cap || []).concat([sqName(land) + (q.o === 0 ? 'M' : 'K')]);
        ev.push({ type: 'kill', pid: q.id, at: land, by: me });
      } else if (q.t === 'q') {
        q.captured = true;
        st.kills[me]++;
        lg.k++; lg.cap = (lg.cap || []).concat(['Q']);
        ev.push({ type: 'kill', pid: q.id, at: land, by: me });
        st.winner = me; st.winReason = 'dronning';
      }
    }
    // Nesseby: fyrste soldat som når MOTSTANDARENS heimrad kaprar kongen.
    // (Kongen har då ingen eigar enno — landingskapring var stengd.)
    if (st.nesseby && p.t === 's') {
      var kk = king(st);
      if (kk && kk.o == null) {
        var lr = pt(land), enemyHome = (me === 0 ? 2 : 0);   // menn -> rad 3, koner -> rad 1
        if (lr.row === enemyHome) {
          kk.o = me;
          st.kings += 1;
          ev.push({ type: 'king', by: me, at: land, nesseby: true });
        }
      }
    }
    // seier: utryddelse. Lágesvuotna-reglene kjenner BARE to vinnings-
    // vilkår: (1) motstanderen har ingen soldater igjen (uansett konge),
    // (2) motstanderens dronning er slått ut. Ikke kongeritt på hjemrad.
    var foe = 1 - me;
    if (st.winner == null) {
      if (!aliveSoldiers(st, foe).length) { st.winner = me; st.winReason = 'utryddelse'; }
    }
    if (st.winner != null) { st.phase = 'over'; return ev; }
    if (diceLeft(st) === 0) endTurn(st, ev);
    return ev;
  }
  function aliveSoldiers(st, o) {
    return soldiers(st, o).filter(function (s) { return !s.captured; });
  }
  function endTurn(st, ev) {
    st.dice = []; st.used = []; st.diceBy = null;
    closePly(st);
    st.turn = 1 - st.turn;
    st.phase = 'roll';
    ev.push({ type: 'turn', next: st.turn });
  }
  function passRemaining(st) {
    var ev = [];
    endTurn(st, ev);
    return ev;
  }
  // Kjøretur etter hvert trekk: kast om X? tapt tur? slutt?
  // returnerer { over, reroll, moves, turnEnded, events }
  function settle(st) {
    if (st.winner != null) return { over: true };
    // Viktig: applyMove kallar allereie endTurn når siste terning er brukt
    // (turn snudd, phase='roll'). Da må NOT settle passere ein gong til —
    // det ville snudd turen tilbake og ete motspelaren sin tur (CPU-en
    // kasta i det uendelege). Fasen seier oss alt: ikkje 'move' = turen
    // er allereie avslutta; berre meld frå om det.
    if (st.phase !== 'move') return { turnEnded: true, events: [] };
    var rk = activeRank(st);
    if (rk == null) return { turnEnded: true, events: passRemaining(st) };
    var lm = legalMoves(st, st.turn);
    if (lm.length === 0) {
      if (rk === RANK.X) return { reroll: true, moves: [] };
      return { turnEnded: true, events: passRemaining(st) };
    }
    return { moves: lm };
  }

  // ---------- serialisering ----------
  function serialize(st) {
    return JSON.stringify({
      v: st.v, turn: st.turn, phase: st.phase, dice: st.dice, used: st.used,
      diceBy: st.diceBy, log: st.log || [],
      winner: st.winner, winReason: st.winReason, queenMode: st.queenMode,
      kings: st.kings, kills: st.kills, moves: st.moves, lastMove: st.lastMove,
      pieces: st.pieces.map(function (p) {
        return { id: p.id, t: p.t, o: p.o, seq: p.seq, at: p.at, released: !!p.released, captured: !!p.captured };
      })
    });
  }
  function deserialize(s) {
    var o = typeof s === 'string' ? JSON.parse(s) : s;
    return {
      v: o.v, turn: o.turn, phase: o.phase, dice: o.dice, used: o.used,
      diceBy: o.diceBy != null ? o.diceBy : o.turn, log: o.log || [],
      winner: o.winner, winReason: o.winReason, queenMode: o.queenMode,
      kings: o.kings, kills: o.kills, moves: o.moves, lastMove: o.lastMove || null,
      pieces: o.pieces.map(function (p) { return { id: p.id, t: p.t, o: p.o, seq: p.seq, at: p.at, released: p.released, captured: p.captured }; })
    };
  }

  var SS = {
    N: N, DICE: DICE, RANK: RANK, CASTLE: CASTLE, TOUR1: TOUR1,
    pt: pt, idx: idx, name: name, rowLine: rowLine, mod: mod,
    tour2Of: tour2Of, soldierAt: soldierAt, startT: startT,
    pathFor: pathFor, moveDest: moveDest, canLand: canLand, soldierLandingOk: soldierLandingOk,
    newGame: newGame, occ: occ, piece: piece, soldiers: soldiers, queen: queen,
    king: king, isHome: isHome, atIdx: atIdx, aliveSoldiers: aliveSoldiers,
    sqName: sqName,
    rollDice: rollDice, openingRoll: openingRoll, roll: roll,
    logPly: logPly, closePly: closePly, logReroll: logReroll,
    activeRank: activeRank, activeValue: activeValue, diceLeft: diceLeft,
    rerollX: rerollX, canRerollX: canRerollX, legalMoves: legalMoves,
    applyMove: applyMove, settle: settle, passRemaining: passRemaining,
    serialize: serialize, deserialize: deserialize
  };
  global.SS = SS;
  if (typeof module !== 'undefined' && module.exports) module.exports = SS;
})(typeof globalThis !== 'undefined' ? globalThis : this);

/* S-UCI FEN — one-line encoding of a full sáhkku state (spec: docs/SAHKKU-UCI.md §6)
 * Attached to SS as SS.toFEN / SS.fromFEN / SS.sqOfName.
 *
 * FEN = "<board> <side> <phase> <dice> <used> <kingSq> <kills> <moves> <result>"
 *
 * board: 3 rows top-to-bottom (row3 = giera home, row1 = menn home), 15 squares
 *        each A..O left-to-right; rows joined '/', squares '-', stacks '+',
 *        empty '.'.
 * tokens: M/m released/home menn soldier · G/g released/home giera soldier
 *         Q/q released/home (L2) menn queen · D/d released/home (D2) giera queen
 *         N king (neutral)
 * dice:   X, 2 (=II), 3 (=III), _ (blank), '-' when not rolled
 * used:   three 0/1 chars marking dice already spent
 */
(function (global) {
  'use strict';
  var SS = global.SS;
  var ENC = { X: 'X', II: '2', III: '3', '_': '_' };
  var DEC = { X: 'X', '2': 'II', '3': 'III', '_': '_' };

  // square -> 60-cycle position per side (row-2 squares sit on TWO passes:
  // first pass = plain, second pass = prime), and home square -> column seq
  var SQ2T = [], SQ2T2 = [], HOME2SEQ = [];
  (function () {
    for (var pl = 0; pl < 2; pl++) {
      var m = [], m2 = [], h = [], t, c, sq, i;
      for (i = 0; i < SS.N; i++) { m.push(-1); m2.push(-1); h.push(-1); }
      for (t = 0; t < 60; t++) {
        sq = SS.soldierAt(pl, t);
        if (m[sq] === -1) m[sq] = t; else if (m2[sq] === -1) m2[sq] = t;
      }
      for (c = 0; c < 15; c++) { sq = SS.soldierAt(pl, SS.startT(pl, c)); h[sq] = c; }
      SQ2T.push(m); SQ2T2.push(m2); HOME2SEQ.push(h);
    }
  })();

  function sqOfName(n) {
    n = String(n).toUpperCase();
    for (var i = 0; i < SS.N; i++) if (SS.name(i) === n) return i;
    return -1;
  }

  function tokenFor(p) {
    if (p.t === 's') {
      if (!p.released) return p.o === 0 ? 'm' : 'g';
      var sq = SS.soldierAt(p.o, p.at);
      // row-2 squares sit on two cycle passes: second pass gets a prime
      var prime = (SQ2T2[p.o][sq] === SS.mod(p.at, 60));
      var base = p.o === 0 ? 'M' : 'G';
      return base + (prime ? "'" : '');
    }
    if (p.t === 'q') return p.o === 0 ? (p.released ? 'Q' : 'q') : (p.released ? 'D' : 'd');
    // king: N neutral · NM seized by menn · NK seized by giera
    return p.o === 0 ? 'NM' : p.o === 1 ? 'NK' : 'N';
  }

  function toFEN(st) {
    var rows = [[], [], []], c, row, k, i;
    for (row = 0; row < 3; row++) for (c = 0; c < 15; c++) rows[row].push([]);
    for (k = 0; k < st.pieces.length; k++) {
      var p = st.pieces[k];
      if (p.captured) continue;
      var pt = SS.pt(SS.atIdx(st, p));
      rows[pt.row][pt.col].push(tokenFor(p));
    }
    var board = [];
    for (row = 2; row >= 0; row--) {
      var cells = [];
      // Kanonisk stabelrekkjefylgje (same som den Python-sida): "'" sorterer
      // etter bokstav, so M < M' < N < NM < NK < Q. Utan dette gav same
      // stilling ulik FEN-string avhengig av kva rekkjefylgje brikkene låg i.
      for (c = 0; c < 15; c++) cells.push(rows[row][c].length
        ? rows[row][c].slice().sort(function (a, b) {
            var ka = a.replace(/'/g, '~'), kb = b.replace(/'/g, '~');
            return ka < kb ? -1 : (ka > kb ? 1 : 0);
          }).join('+')
        : '.');
      board.push(cells.join('-'));
    }
    var side = st.turn === 0 ? 'w' : 'b';
    var dice = (st.dice && st.dice.length === 3) ? st.dice.map(function (d) { return ENC[d]; }).join('') : '-';
    var used = (st.used && st.used.length === 3) ? st.used.map(function (u) { return u ? '1' : '0'; }).join('') : '000';
    return board.join('/') + ' ' + side + ' ' + st.phase + ' ' + dice + ' ' + used +
      ' ' + SS.name(SS.king(st).at) + ' ' + st.kills[0] + ':' + st.kills[1] +
      ' ' + st.moves + ' ' + (st.winner == null ? '*' : st.winner === 0 ? '1-0' : '0-1');
  }

  function parseToken(tok, st, sq, assigned) {
    var pl, rel, i, p;
    if (tok === 'N' || tok === 'NM' || tok === 'NK') {
      var k = SS.king(st);
      k.at = sq;
      k.o = tok === 'NM' ? 0 : tok === 'NK' ? 1 : null;
      return true;
    }
    if (tok === 'M' || tok === 'm' || tok === 'G' || tok === 'g' ||
        tok === "M'" || tok === "G'") {
      var prime = tok.length > 1;
      var baseTok = prime ? tok[0] : tok;
      pl = (baseTok === 'M' || baseTok === 'm') ? 0 : 1;
      rel = (baseTok === 'M' || baseTok === 'G') || prime;
      var t = rel ? (prime ? SQ2T2[pl][sq] : SQ2T[pl][sq]) : SS.startT(pl, HOME2SEQ[pl][sq]);
      if (rel && t === -1) throw new Error('fen: open soldat «' + tok + '» på ' + SS.name(sq) + ' står utanfor ' + (pl ? 'giera' : 'menn') + ' si tallsyklus' + (prime ? ' (andre pass)' : ''));
      if (!rel && HOME2SEQ[pl][sq] === -1) throw new Error('fen: heimemann «' + tok + '» står ikje på heimrekka (' + SS.name(sq) + ')');
      for (i = 0; i < st.pieces.length; i++) {
        p = st.pieces[i];
        if (p.t !== 's' || p.o !== pl || assigned[p.id]) continue;
        p.released = rel; p.at = t; p.captured = false;
        assigned[p.id] = true;
        return true;
      }
      throw new Error('fen: for mange soldatar for ' + (pl ? 'giera' : 'menn'));
    }
    if (tok === 'Q' || tok === 'q' || tok === 'D' || tok === 'd') {
      pl = (tok === 'Q' || tok === 'q') ? 0 : 1;
      rel = (tok === 'Q' || tok === 'D');
      for (i = 0; i < st.pieces.length; i++) {
        p = st.pieces[i];
        if (p.t === 'q' && p.o === pl) {
          p.captured = false; p.released = rel;
          if (rel) p.at = sq;   // home queen is encoded at her start square
          return true;
        }
      }
      throw new Error('fen: ukjend dronning-token «' + tok + '»');
    }
    throw new Error('fen: ukjend brikke «' + tok + '» på ' + SS.name(sq));
  }

  function fromFEN(fen) {
    var w = String(fen).trim().split(/\s+/);
    if (w.length < 9) throw new Error('fen: for få felt (' + w.length + '/9)');
    var st = SS.newGame({});
    var rows = w[0].split('/');
    if (rows.length !== 3) throw new Error('fen: board må ha 3 rekker');
    var assigned = {}, rowNames = [2, 1, 0], r, c;
    for (r = 0; r < 3; r++) {
      var cells = rows[r].split('-');
      if (cells.length !== 15) throw new Error('fen: rekkje ' + r + ' har ' + cells.length + ' felt (15 krävst)');
      for (c = 0; c < 15; c++) {
        if (cells[c] === '.') continue;
        cells[c].split('+').forEach(function (tok) { parseToken(tok, st, SS.idx(c, rowNames[r]), assigned); });
      }
    }
    st.pieces.forEach(function (p) {
      if (p.t === 's' && !assigned[p.id]) p.captured = true;
      if (p.t === 'q') {
        var seenTok = false;
        for (var i = 0; i < w[0].length; i++) {
          var ch = w[0][i];
          if ((p.o === 0 && (ch === 'Q' || ch === 'q')) || (p.o === 1 && (ch === 'D' || ch === 'd'))) { seenTok = true; break; }
        }
        if (!seenTok) p.captured = true;
      }
    });
    if (w[1] !== 'w' && w[1] !== 'b') throw new Error('fen: side «' + w[1] + '» (w/b)');
    st.turn = w[1] === 'w' ? 0 : 1;
    if (['roll', 'move', 'over'].indexOf(w[2]) < 0) throw new Error('fen: fase «' + w[2] + '»');
    st.phase = w[2];
    st.dice = w[3] === '-' ? [] : (function () {
      var d = w[3].split('');
      if (d.length !== 3) throw new Error('fen: terningar «' + w[3] + '»');
      return d.map(function (ch) {
        if (!DEC[ch]) throw new Error('fen: terning «' + ch + '»');
        return DEC[ch];
      });
    })();
    if (!/^[01]{3}$/.test(w[4])) throw new Error('fen: brukt-streng «' + w[4] + '»');
    st.used = st.dice.length ? w[4].split('').map(function (ch) { return ch === '1'; }) : [];
    var ks = sqOfName(w[5]);
    if (ks < 0) throw new Error('fen: konge-felt «' + w[5] + '»');
    SS.king(st).at = ks;
    var kl = w[6].split(':');
    if (kl.length !== 2 || !/^\d+$/.test(kl[0]) || !/^\d+$/.test(kl[1])) throw new Error('fen: slag «' + w[6] + '»');
    st.kills = [parseInt(kl[0], 10), parseInt(kl[1], 10)];
    st.moves = parseInt(w[7], 10) || 0;
    if (w[8] === '1-0') { st.winner = 0; st.phase = 'over'; }
    else if (w[8] === '0-1') { st.winner = 1; st.phase = 'over'; }
    else if (w[8] !== '*') throw new Error('fen: resultat «' + w[8] + '»');
    return st;
  }

  SS.toFEN = toFEN;
  SS.fromFEN = fromFEN;
  SS.sqOfName = sqOfName;
  global.SS = SS;
})(typeof globalThis !== 'undefined' ? globalThis : this);

var SS = (typeof globalThis !== 'undefined' ? globalThis : this).SS;

// ==================== FEATURES + ACTIONS (mirrors features.py) ============
var N_PLANES = 16, N_GLOBAL = 16;   // fylt inn av export.py
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
var ENGINE_NAME = 'SahkkuNN 1.0';   // settast ved eksport (--name)
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
  NN.exported.loadWeights(JSON.parse(fs.readFileSync(path.join(__dirname, 'nais4-nn.weights.json'), 'utf8')));
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
