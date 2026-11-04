"""Testar materiell-verdimodulen: POV-forlegingsnad, dronningrisiko
skalert etter Lars sine reglar (seljarantal, tettleik, konge i Near),
fiendeleg dronning som gevinst, clamp og justerbarheit."""
from __future__ import annotations

from sahkku_ai import engine as E
from sahkku_ai.materials import (CASTLE, MATERIALS, blended_value,
                                 material_sum, parse_material_option)


def _st(side=0):
    st = E.State(first=0)
    st.turn = side
    return st


def main():
    w = dict(MATERIALS)

    # 1) startstillinga: symmetrisk -> forlegingsnad 0 frå begge sider
    assert abs(material_sum(_st(0), w)) < 1e-9, material_sum(_st(0), w)
    assert abs(material_sum(_st(1), w)) < 1e-9, material_sum(_st(1), w)

    # 2) å ta ein fiendesoldat (fjernast) er GEVINST for oss
    st = _st(0)
    foe_soldier = E.soldiers(st, 1)[0]
    foe_soldier.captured = True
    assert material_sum(st, w) > 0, "kapra fiendesoldat må heve summen"

    # 3) vi mistar eigen soldat -> summen går ned
    st = _st(0)
    E.soldiers(st, 0)[0].captured = True
    assert material_sum(st, w) < 0, "tapt eigen soldat må senke summen"

    # 4) dronningrisiko SKALERT: eiga utløyst dronning straffast hardare
    #    med FLEIRE eigne soldatar (Lars: ikkje aktiver med >3 soldatar)
    st = _st(0)
    E.queen(st, 0).released = True
    few = dict(w, qActPerS=-0.2)          # bratt stigning per soldat
    sum_dense = material_sum(st, few)     # 15 eigne soldatar på brettet
    for p in E.soldiers(st, 0)[:12]:      # fjern 12 av våre -> berre 3 att
        p.captured = True
    sum_sparse = material_sum(st, few)
    assert sum_sparse > sum_dense, (sum_dense, sum_sparse)

    # 5) kongen langt frå slottet gjer aktiveringa endå farlegare
    st1, st2 = _st(0), _st(0)
    E.queen(st1, 0).released = True
    E.queen(st2, 0).released = True
    E.king(st1).o = 0                     # vi kontrollerer kongen
    E.king(st2).o = 0
    E.king(st2).at = 7                    # A1 — langt frå H2 (22)
    assert material_sum(st2, w) < material_sum(st1, w), "konge borte = meir risiko"
    # 5b) heilt MANGLANDE eigen konge (slått ut) = verst
    st3 = _st(0)
    E.queen(st3, 0).released = True
    E.king(st3).captured = True
    assert material_sum(st3, w) <= material_sum(st2, w), "utan konge = maks risiko"

    # 6) FIENDEN sin utløyste dronning er GEVINST for oss (prisen på ho)
    st = _st(0)
    E.queen(st, 1).released = True
    assert material_sum(st, w) > 0, "utløyst fiendedronning må telast som gevinst"

    # 7) blending: clamp [-1,1] og monotont i materialet
    assert -1.0 <= blended_value(0.5, _st(), w) <= 1.0
    assert blended_value(0.9, _st(), w) >= blended_value(0.1, _st(), w)
    assert blended_value(5.0, _st(), w) == 1.0
    assert blended_value(-5.0, _st(), w) == -1.0

    # 8) setoption-parser: berre kjende nøklar, talla tolerast
    got = parse_material_option("sHome=0.04, qActPerS=-0.12 ; blend=0.5, snusk=99, kFar=abc")
    assert got["sHome"] == 0.04 and got["qActPerS"] == -0.12 and got["blend"] == 0.5
    assert got["king"] == MATERIALS["king"], "ukjend/ugyldig felt må ikkje endrast"
    assert "snusk" not in got

    print("MATERIALS TEST OK: null i start, kapring=gevinst, dronningrisiko "
          "skalert (soldatar/tettleik/konge), fiendedronning=premie, clamp, parser")


if __name__ == "__main__":
    main()
