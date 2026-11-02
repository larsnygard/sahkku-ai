"""Testar materiell-verdimodulen: aktiverte vs heimeståande brikker,
net-paritet for blend-blad, og justerbarheit (Lars: verdiane skal
kunne finjusterast etterpå)."""
from __future__ import annotations

from sahkku_ai import engine as E
from sahkku_ai.materials import MATERIALS, blended_value, material_sum, parse_material_option


def _st():
    st = E.State(first=0)
    return st


def main():
    st = _st()
    w = dict(MATERIALS)
    base = material_sum(st, w)
    # startstillinga: 15+15 soldatar heime, 2 dronningar heime, 1 konge
    s_home = 30 * w["sHome"] + 2 * w["qHome"] + w["king"]
    assert abs(base - s_home) < 1e-9, (base, s_home)

    # utløyst soldat tel meir enn heimeståande
    w2 = dict(w, sAct=w["sHome"] + 0.1)
    st2 = _st()
    p = E.soldiers(st2, 0)[0]
    p.released = True
    assert material_sum(st2, w2) > material_sum(st2, w), "utløyst soldat må telast meir"

    # AKTIVERT dronning er FLEMM: qAct negativ, qHome positiv
    assert w["qAct"] < 0 < w["qHome"], "aktivert dronning må straffast"
    st3 = _st()
    q = E.queen(st3, 0)
    q.released = True
    d = material_sum(st3, w) - (s_home)
    assert d < 0, f"aktivert dronning må senke summen (feil: {d})"

    # kongen straffast for å gå frå slottet
    st4 = _st()
    k = E.king(st4)
    k.at = 7   # A1 — langt frå H2
    assert material_sum(st4, w) < s_home, "kongen bøteleggjast for å forlate slottet"

    # blending: clamp [-1,1] og monotont i materialet
    v = blended_value(0.5, _st(), w)
    assert -1.0 <= v <= 1.0
    assert blended_value(0.9, _st(), w) >= blended_value(0.1, _st(), w)
    assert blended_value(5.0, _st(), w) == 1.0
    assert blended_value(-5.0, _st(), w) == -1.0

    # setoption-parser: berre kjende nøklar, talla tolerast
    got = parse_material_option("sHome=0.04, qAct=-0.12 ; blend=0.5, snusk=99, kFar=abc")
    assert got["sHome"] == 0.04 and got["qAct"] == -0.12 and got["blend"] == 0.5
    assert got["king"] == MATERIALS["king"], "ukjend/ugyldig felt må ikkje endrast"
    assert "snusk" not in got

    print("MATERIALS TEST OK: aktivert≠heime (soldat/dronning), konge-bøte, "
          "clamp, justerbar parser")


if __name__ == "__main__":
    main()
