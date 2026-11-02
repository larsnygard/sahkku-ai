"""Materiell stillingsverdi — justerbare vekter for search-blending.

Verdienettet seier kven som truleg vinn; materialet seier kva som står på
brettet VERKANDES: ein utløyst soldat er meir verdt enn ein heimeståande,
og ei AKTIVERT dronning er ein FLEMM (ho kan takast, og tapet er umiddelbart
når ho fell). Desse tala er ein manuell prioritet som blandast inn i
verdinettet i bladsøkjinga (scorefen/alpha-beta), IKKJE i treningen.

Alle vekter er eitt lite sett som kan justerast etterpå:
  * Python:   MATERIALS (dict) — tools og testar brukar det direkte
  * JS-motoren: MAT (dict) i runner_template.js — S-UCI `setoption name
    Material value sHome=0.04,sAct=0.06,qHome=0.05,qAct=-0.12,blend=0.4`
    endrar dei i sanntid; `material`-kommandoen skriv dei noverande.

Blendinga (same formel i Python og JS):

    blended = clamp(net_value + blend * material_sum, -1, 1)

der `net_value` er verdinettet sitt tal frå «sidan som skal trekke»-syn,
og `material_sum` er summen av brikkeverdiane under.
"""
from __future__ import annotations

# Standardvektar (Lars godkjende 2026-11-02; justerbare):
#  sHome  heimeståande soldat        — trygg, latent mobilitet
#  sAct   utløyst soldat             — aktiv kraft (risikoen ligg i trussel-talet)
#  qHome  heimeståande dronning      — trygt potensiale
#  qAct   aktivert dronning          — FLAMM: kan takast, tapet er umiddelbart
#  king   kongen                     — bufferen: gjev alltid eit trekk
#  kFar   straff per felt kongen er frå slottet (H2) — «hald kongen attende»
#  blend  kor mykje materialet veg mot verdinettet i bladverdien
MATERIALS = {
    "sHome": 0.03,
    "sAct": 0.05,
    "qHome": 0.04,
    "qAct": -0.10,
    "king": 0.02,
    "kFar": -0.02,
    "blend": 0.30,
}

CASTLE = 22


def material_sum(st, w: dict | None = None) -> float:
    """Summen av brikkeverdiane (side-nøytralt: begge sider telst)."""
    w = w or MATERIALS
    from . import engine as E
    from .features import _cheb
    s = 0.0
    for p in st.pieces:
        if p.captured:
            continue
        if p.t == "s":
            s += w["sAct"] if p.released else w["sHome"]
        elif p.t == "q":
            s += w["qAct"] if p.released else w["qHome"]
        elif p.t == "k":
            s += w["king"] + w["kFar"] * (min(_cheb(p.at, CASTLE), 14) / 14.0)
    return s


def blended_value(net_value: float, st, w: dict | None = None) -> float:
    """Bladverdi: net + blend·materiale, clampa til [-1, 1]."""
    w = w or MATERIALS
    v = net_value + w["blend"] * material_sum(st, w)
    return max(-1.0, min(1.0, v))


def parse_material_option(s: str) -> dict:
    """`setoption name Material value sHome=0.04,qAct=-0.12` -> dict (berre kjende nøklar)."""
    out = dict(MATERIALS)
    for part in s.replace(";", ",").split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k in MATERIALS:
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out
