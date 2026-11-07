/* SPG — «Sáhkku Portable Game» — PGN-inspirert notasjon for sáhkku.
 * Det finst ingen etablert sáhkku-notasjon (verken Ludii, Wikipedia eller
 * regeldokumenta har definert ei), difor denne: tekstfil, éi linje per
 * kast, med terningkast (også X-omkast), flytt og landingseffekt.
 * FULL SPESIFIKASJON: docs/SPG.md.
 *
 *   [Event "Kapplanginga 3. rundar"]        PGN-style header
 *   [Date "2026-10-25"]
 *   [White "Menn (bieljet)"]
 *   [Black "Koner (giera)"]
 *   [Result "1-0"]                            1-0 M · 0-1 K · * ugjort
 *   [Termination "dronning"]                  dronning/utryddelse/overgivelse
 *   SPG-V1
 *
 *   1. open: M(X X X) K(III III III) -> M   # opningsritual: kappkast, vinnar M
 *   1. M: II II III                         # kvart kast: kastNr. side: terningar
 *   2. K: X _ III                           # (omkast #n) markerer omkast-kast
 *      X->A2                                 #   X-løysning (mål-felt identifiserer brikka)
 *      III: QA2->F2                          #   II:/III: = terningsledd (X-ledd har ikje prefiks)
 *   3. M: III III III
 *      III: KA2->D2                          #   soldatar flyttar FRAMOVER i tallsyklusen
 *      III: KD2->G2                          #   same brikke, neste ledd
 *   8. K: X II _
 *      KG2->H2§                              #   § = kongen teken på tårnet (midtre rekkje)
 *      II: KH2->J2
 *   13. M: III III II
 *       II: QI2->G2x [Q]                     #   x = landing tok brikker, [felt+eigar / Q]
 *   1-0 {dronning}                           # sluttlinje: resultat {orsak}
 *
 * Bokstav: M=bielje (menn), K=giera (konar), Q=dronninga, k=kongen.
 * Felt: kolonne A–O, rekkje 1–3. Alle felt namngjevne frå Menn si side.
 */
'use strict';
(function (global) {
  var SS = global.SS || global.SahkkuCore;

  function diceStr(d) { return d.join(' '); }
  function letter(t, o) { return t === 's' ? (o === 0 ? 'M' : 'K') : t === 'q' ? 'Q' : 'k'; }

  function toSPG(st, meta) {
    meta = meta || {};
    var L = [];
    L.push('[Event "' + (meta.event || 'Sáhkku-parti') + '"]');
    // SPG 1.2: tidspunktet partiet BYRJA, ikkje då det vart eksportert.
    // Date/Time er lokal tid (som PGN sin [Time]); UTC gjer augneblinken
    // entydig, so han kan sorterast og samanliknast på tvers av tidssoner.
    var start = meta.start ? new Date(meta.start) : new Date();
    var pad2 = function (n) { return (n < 10 ? '0' : '') + n; };
    L.push('[Date "' + (meta.date || (start.getFullYear() + '-' +
      pad2(start.getMonth() + 1) + '-' + pad2(start.getDate()))) + '"]');
    L.push('[Time "' + (meta.time || (pad2(start.getHours()) + ':' +
      pad2(start.getMinutes()) + ':' + pad2(start.getSeconds()))) + '"]');
    L.push('[UTC "' + (meta.utc || start.toISOString().replace(/\.\d+Z$/, 'Z')) + '"]');
    // SPG 1.1: sídene heiter Menn og Koner (ikkje White/Black). Namn på
    // forma «Etternamn, Fornamn». Ukjende namn: dei samiske side-namna
    // (olbmát = menn, gálgut = koner). Datamaskin: «(c) <motor>».
    L.push('[Menn "' + (meta.menn || meta.white || 'Olbmát') + '"]');
    L.push('[Koner "' + (meta.koner || meta.black || 'Gálgut') + '"]');
    L.push('[Result "' + (st.winner === 0 ? '1-0' : st.winner === 1 ? '0-1' : '*') + '"]');
    if (st.winReason) L.push('[Termination "' + st.winReason + '"]');
    // Regelvarianten må stå i fila: utan han kan ein Nesseby-fil ikkje spelast
    // opp rett att (kongen byter eigar under dei vanlege reglane).
    if (st.nesseby) L.push('[Variant "Nesseby"]');
    L.push('SPG-V1');
    L.push('');
    // SPG 1.2: kommentarar, same syntaks som PGN ({…} og ; til linjeslutt).
    var cm = meta.comment ? (Array.isArray(meta.comment) ? meta.comment : [meta.comment]) : [];
    cm.forEach(function (c) { L.push('{' + String(c) + '}'); });
    if (cm.length) L.push('');
    var plyNo = 0;
    (st.log || []).forEach(function (ply) {
      if (ply.t === 'open') {
        ply.rolls.forEach(function (r, ix) {
          var last = ix === ply.rolls.length - 1;
          var x = function (d) { return d.filter(function (v) { return v === 'X'; }).length; };
          L.push('1. open' + (ply.rolls.length > 1 ? '#' + (ix + 1) : '') +
            ': M(' + diceStr(r.p0) + ') K(' + diceStr(r.p1) + ')' +
            (last ? ' -> ' + (x(r.p0) > x(r.p1) ? 'M' : 'K') : ''));
        });
        return;
      }
      var side = ply.t === 0 ? 'M' : 'K';
      ply.roll.forEach(function (d, ix) {
        plyNo++;
        L.push(plyNo + '. ' + side + (ix > 0 ? ' (omkast #' + ix + ')' : '') + ': ' + diceStr(d));
        if (ix > 0) L.push('   X!');
      });
      ply.mv.forEach(function (m) {
        var head = letter(m.t, m.o);
        var tail = m.kg ? '§' : (m.k > 0 ? 'x [' + (m.cap || []).join(',') + ']' : '');
        // die prefix: II/III moves carry their face (m.v is NUMERIC: 1=X, 2, 3);
        // X-slot moves (v=1) and releases (X->) carry none
        var pre = (m.v === 2) ? 'II: ' : (m.v === 3) ? 'III: ' : '';
        if (m.rel) L.push('   X->' + SS.sqName(m.to));
        else L.push('   ' + pre + head + SS.sqName(m.from) + '->' + SS.sqName(m.to) + tail);
      });
      // valfri kommentar på turen (t.d. sett inn av eit analyseverktøy)
      if (ply.c) L.push('   {' + String(ply.c) + '}');
    });
    if (st.winner !== null) L.push((st.winner === 0 ? '1-0' : '0-1') + ' {' + (st.winReason || '') + '}');
    return L.join('\n') + '\n';
  }

  // =======================================================================
  // Lesing (SPG 1.2). Appen kunne tidlegare berre SKRIVE SPG — no kan ho
  // lese, so arkivet tek imot .spg og .spgz utan omveg om JSON.
  // =======================================================================

  var TAG = /^\[(\w+)\s+"(.*)"\]\s*$/;
  var OPEN = /^1\.\s+open(?:#(\d+))?:\s*M\((X|II|III|_)\s+(X|II|III|_)\s+(X|II|III|_)\)\s+K\((X|II|III|_)\s+(X|II|III|_)\s+(X|II|III|_)\)\s*(?:->\s*([MK]))?\s*$/;
  // SPG 1.2/§4.2: omkast-merket står ETTER sida og før kolon —
  // «10. M (omkast #1): II _ X». Vi tek òg den hale-forma nokon skreiv, so
  // gamle filer ikkje brest.
  var ROLL = /^(\d+)\.\s+([MK])(?:\s*\(omkast\s*#(\d+)\))?:\s+(X|II|III|_)\s+(X|II|III|_)\s+(X|II|III|_)(?:\s*\(omkast\s*#(\d+)\))?\s*$/;
  var MOVE = /^(?:(II|III):\s*)?([MkKQ])([A-O][123])->([A-O][123])(§|x \[[^\]]*\])?\s*$/;
  var REL = /^X->([A-O][123])\s*$/;
  var END = /^(1-0|0-1|\*)\s*(?:\{(\w+)\})?\s*$/;
  var XMARK = /^X!\s*$/;

  // Kommentarar: {…} (kan gå over fleire linjer, ingen nesting) og ; til
  // linjeslutt. Resultatlinja «1-0 {dronning}» har {…} som ÅRSAK, ikkje
  // kommentar, so ho vert halden utanfor.
  function stripComments(text) {
    var found = [], PROT = '\u0000SPGPROT\u0000', protectedLines = [], lines = [];
    text.split(/\r?\n/).forEach(function (raw) {
      var s = raw.trim();
      if (/^(1-0|0-1|\*)\s*\{[^{}]*\}$/.test(s)) { protectedLines.push(s); lines.push(PROT); }
      else lines.push(raw);
    });
    var body = lines.join('\n').replace(/\{([^{}]*)\}/g, function (m, c) {
      found.push(c.trim()); return ' ';
    });
    var out = [];
    body.split('\n').forEach(function (raw) {
      if (raw.trim() === PROT) { out.push(protectedLines.shift() || ''); return; }
      var cut = raw.indexOf(';');
      if (cut >= 0) {
        var note = raw.slice(cut + 1).trim();
        if (note) found.push(note);
        raw = raw.slice(0, cut);
      }
      out.push(raw);
    });
    return { text: out.join('\n'), comments: found };
  }

  function parseOne(text) {
    var st = stripComments(text);
    var tags = {}, events = [], result = null, reason = '', bad = null, nesseby = false;
    st.text.split('\n').forEach(function (raw) {
      var line = raw.trim();
      if (!line) return;
      var m = TAG.exec(line);
      if (m) {
        tags[m[1]] = m[2];
        // [Variant "Nesseby"] — same namnet som skrivaren brukar
        if (/^variant$/i.test(m[1]) && /nesseby/i.test(m[2])) nesseby = true;
        return;
      }
      if (line === 'SPG-V1') return;
      m = OPEN.exec(line);
      if (m) {
        if (m[8]) events.push({ k: 'open', first: m[8] === 'M' ? 0 : 1 });
        else events.push({ k: 'open', first: 0 });
        return;
      }
      m = ROLL.exec(line);
      if (m) {
        var reroll = m[3] || m[7];
        events.push({ k: reroll ? 'reroll' : 'roll', faces: [m[4], m[5], m[6]],
                      side: m[2] === 'M' ? 0 : 1 });
        return;
      }
      m = MOVE.exec(line);
      if (m) { events.push({ k: 'move', letter: m[2], from: m[3], to: m[4], die: m[1] || null }); return; }
      m = REL.exec(line);
      if (m) { events.push({ k: 'rel', to: m[1] }); return; }
      if (XMARK.test(line)) return;
      m = END.exec(line);
      if (m) { result = m[1]; reason = m[2] || ''; return; }
      if (!bad) bad = line;
    });
    return { tags: tags, events: events, result: result, reason: reason,
             comments: st.comments, bad: bad, nesseby: nesseby };
  }

  // Eitt eller fleire parti i same fil (som PGN): eit nytt parti byrjar på
  // ei tag-linje når det førre alt har hatt SPG-V1-merket.
  function splitGames(text) {
    var lines = text.split(/\r?\n/), out = [], cur = [], seenMarker = false, inBrace = false;
    lines.forEach(function (raw) {
      var t = raw.trim();
      // hald styr på {…} so ikkje «[Event» inne i ein kommentar byrjar eit parti
      var opens = (t.match(/\{/g) || []).length, closes = (t.match(/\}/g) || []).length;
      var startsTag = !inBrace && TAG.test(t);
      if (startsTag && seenMarker) { out.push(cur.join('\n')); cur = []; seenMarker = false; }
      if (!inBrace && t === 'SPG-V1') seenMarker = true;
      cur.push(raw);
      if (opens > closes) inBrace = true;
      else if (closes > opens) inBrace = false;
    });
    if (cur.join('').trim()) out.push(cur.join('\n'));
    return out;
  }

  function parseMany(text) {
    return splitGames(text).map(parseOne)
      .filter(function (g) { return g.events.length || Object.keys(g.tags).length; });
  }

  // --- trekk-attkjenning (spegl av sahkku_ai/spg.py) ---------------------
  function pieceLetter(p) {
    if (p.t === 'q') return 'Q';
    if (p.t === 'k') return 'k';
    return p.o === 0 ? 'M' : 'K';
  }
  function destOf(st, mv) {
    var p = SS.piece(st, mv.pid);
    if (p.t === 's') return SS.soldierAt(p.o, p.at + (mv.kind === 'rel' ? 1 : mv.v));
    return SS.moveDest(st, mv);
  }
  function fromOf(st, mv) {
    var p = SS.piece(st, mv.pid);
    return p.t === 's' ? SS.soldierAt(p.o, p.at) : p.at;
  }

  // Spel eit parsa parti opp att; gjev arkiv-oppføring (steps + fens).
  function toRecord(g, meta) {
    // Regelmodusen må med: under Nesseby kan kongen ikkje kaprast ved landing,
    // so ein Nesseby-fil som vert spela opp med vanlege reglar hamnar i heilt
    // feil stilling (kongen byter eigar) og seinare trekk vert ulovlege.
    var nb = (meta && meta.nesseby != null) ? !!meta.nesseby
           : (g.nesseby != null ? !!g.nesseby : false);
    var st = SS.newGame({ first: 0, nesseby: nb });
    var steps = [], fens = [SS.toFEN(st)], applied = 0, first = null;
    var curStep = null;
    g.events.forEach(function (ev) {
      if (ev.k === 'open') {
        st = SS.newGame({ first: ev.first, nesseby: nb });
        first = ev.first;
        st.turn = ev.first;
        fens = [SS.toFEN(st)];
        return;
      }
      if (ev.k === 'roll' || ev.k === 'reroll') {
        st.turn = ev.side;
        st.dice = ev.faces.slice();
        st.used = ev.faces.map(function (d) { return d === '_'; });
        st.diceBy = st.turn;
        st.phase = 'move';
        if (ev.k === 'roll') { curStep = { t: ev.side, rolls: [ev.faces.slice()], moves: [] }; steps.push(curStep); }
        else if (curStep) curStep.rolls.push(ev.faces.slice());
        return;
      }
      var pool = SS.legalMoves(st, st.turn), pick = null;
      for (var i = 0; i < pool.length; i++) {
        var mv = pool[i];
        if (ev.k === 'rel') {
          if (mv.kind === 'rel' && SS.sqName(destOf(st, mv)) === ev.to) { pick = mv; break; }
        } else {
          var p = SS.piece(st, mv.pid);
          if (pieceLetter(p) !== ev.letter) continue;
          if (SS.sqName(fromOf(st, mv)) !== ev.from) continue;
          if (SS.sqName(destOf(st, mv)) !== ev.to) continue;
          if (ev.die === 'II' && mv.v !== 2) continue;
          if (ev.die === 'III' && mv.v !== 3) continue;
          pick = mv; break;
        }
      }
      if (!pick) throw new Error('SPG: ukjent trekk «' + (ev.k === 'rel' ? 'X->' + ev.to
        : ev.letter + ev.from + '->' + ev.to) + '» i ' + (meta && meta.name ? meta.name : 'fila'));
      var frm = fromOf(st, pick), to = destOf(st, pick);
      SS.applyMove(st, pick);
      applied++;
      fens.push(SS.toFEN(st));
      var step = curStep || (curStep = { t: st.turn, rolls: [], moves: [] });
      step.moves.push([frm, to, pick.v, pick.kind === 'rel' ? 1 : 0]);
    });
    return {
      id: null,
      ts: dateFromTags(g.tags),
      mode: 'spg',
      engine: g.tags.Koner && /\(c\)/.test(g.tags.Koner) ? g.tags.Koner.replace(/^.*\(c\)\s*/, '') : '',
      level: 2,
      first: first === null ? 0 : first,
      nesseby: !!nb,
      result: g.result || '*',
      reason: g.reason || (g.tags.Termination || ''),
      plies: applied,
      steps: steps,
      fens: fens,
      menn: g.tags.Menn || '', koner: g.tags.Koner || '',
      event: g.tags.Event || '', utc: g.tags.UTC || '', time: g.tags.Time || '',
      comments: g.comments || [],
      spg: null,
    };
  }

  function dateFromTags(tags) {
    var d = tags.Date || '', t = tags.Time || '00:00:00';
    if (tags.UTC) return tags.UTC;
    if (/^\d{4}-\d{2}-\d{2}$/.test(d)) return d + 'T' + t;
    return new Date().toISOString();
  }

  // Skriv fleire parti til éi fil (PGN-stil: blank linje mellom).
  function toSPGMany(states, metas) {
    return states.map(function (s, i) { return toSPG(s, metas[i] || {}); }).join('\n');
  }

  global.SPG = { toSPG: toSPG, toSPGMany: toSPGMany, parseOne: parseOne,
                 parseMany: parseMany, toRecord: toRecord, stripComments: stripComments,
                 splitGames: splitGames };
})(typeof window !== 'undefined' ? window : globalThis);
