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
    for (i = 0; i < on.length; i++) {
      q = on[i];
      if (q === p) continue;
      if (p.t === 's') {
        if (q.t === 'k') {
          if (q.o === p.o) return false;               // egen konge: kongen deler ALDRI linje med egne
          continue;                                    // fiendtlig konge: kapre ved landing
        }
        if (q.t === 'q' && q.o !== p.o) continue;      // slå fiendtlig dronning
        if (q.t === 'q' && q.o === p.o) return false;  // egen dronning: nektet
        // soldater: egne tillatt (stable), fiender slås, hjemmebrikker:
        if (isHome(q) && q.o === p.o) return true;     // egen hjemrad: se soldatLandingOk
        if (isHome(q)) return false;                   // inn på fiendes hjemrad: nektet
        return true;
      }
      if (p.t === 'k') {
        // Kongen lander ALDRI på en linje med brikker: han kan ikke slå ut
        // og deler ikke linje med andre. Kapring av kongen skjer bare når
        // EN SOLDAT eller DRONNING lander på han (canLand sin soldat-gren).
        return false;
      }
      if (p.t === 'q') {
        if (q.t === 's' && q.o === p.o) return false;  // deler ikke felt med egne
        if (q.t === 'q' && q.o === p.o) return false;
        if (q.t === 'k') continue;                     // kapre kongen
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
      queenMode: opts.queenMode !== false
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
  function rerollX(st, rnd) {
    if (!canRerollX(st)) return null;
    for (var i = 0; i < 3; i++) if (st.dice[i] === 'X') {
      var nf = rollDice(rnd)[0];
      st.dice[i] = nf;
      st.used[i] = (nf === '_');   // eit nytt blankt fjes er brukt med ein gong
    }
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
    // X (sáhkku) VEKKJER berre: ho løyser ei inaktiv brikke og flyttar henne
    // eitt felt som del av vekkinga. Ho flyttar ALDRI brikker som alt er
    // utløyste (soldat, dronning eller kongen) — det gjer II og III.
    // Regelen er Lars si presisering: «X kan brukes til å vekke ei inaktiv
    // brikke ... og deretter flytte ett felt. Flyttet er en del av vekkinga.»
    if (rk === RANK.X) return out;
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
      } else if (p.t === 'k') {
        if (p.o !== me) continue;
        var km2 = genCycleMoves(st, p, v);
        for (var z2 = 0; z2 < km2.length; z2++) out.push(km2[z2]);
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
