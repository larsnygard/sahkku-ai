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
      for (c = 0; c < 15; c++) cells.push(rows[row][c].length ? rows[row][c].join('+') : '.');
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
