"""Materiell stillingsverdi — POV-DELTE fordeling, justerbare vekter.

Kvifor sideteikn (Lars 2026-11-04): den fyrste versjonen summerte brikke-
verdien for BEGGE sider med same pluss-koeffisient. Det gjer materialet til
eit INVENT-LIST (brettet med 30 soldatar får høg materialsum uansett kven
som eig dei) — ikkje ein FØRELESGNAD. For å priorisere siger ved utsleting
summen må vere ein FORLELESGNAD: kvar brikke har forteikn — våre +, deira −.
Ein kapra fiendesoldat er no ein GEVINST; ei truga dronning er ein premie, ikkje eit tap.

Dronninga si Risiko (Lars sine tre reglar, alle som eigenvektar):
  1. Ei utløyst dronning kan takast — tapet er umiddelbart når ho fell.
  2. Med mange eigne soldatar på brettet er aktivering unødig.
  3. Fleire brikker totalt = fleire angrep = større fare.
  4. Ut kongen i Near (nær slottet) er dronninga nesten umogleg å verne.
  Difor er prisen for å aktivere eiga dronning SKALERT:
    qActBase + qActPerS·(eigne soldatar) + qActDen·(brikker på brettet)
           + qActNoKing·(kongen meir enn 4 felt frå slottet)
  Kvar vekt er justerbar i sanntid utan retraining.

  5. Motstandaren si utløyste dronning er GEVINST for oss (+qActEnemy):
     ho kan takast. Symmetrisk straff (gamle modell) gav engine null grunn
     til å jakte fiendtlege dronningar.

Blendinga (same formel i Python og JS):

    blended = clamp(net_value + blend * material_sum, -1, 1)

der `material_sum` ALLTID er rekna frå `st.turn` si side (POV), og verdien
frå nettallta allereie er det. Summen i startstillinga er 0 (symmetri).
"""
from __future__ import annotations

# Standardvektar (Lars 2026-11-04; justerbare):
#  sHome      heimeståande soldat      — trygg, latent mobilitet
#  sAct       utløyst soldat           — aktiv kraft
#  qHome      heimeståande dronning    — trygt potensiale
#  qActBase   fast kostnad for å aktivere eiga dronning
#  qActPerS   auking per eigen soldat (meir makt = meir treng ikkje dronninga)
#  qActDen    auking per brikke på brettet (tettleik = fare)
#  qActNoKing kostnad når kongen ikkje dekkjer sonen (>4 felt frå slottet)
#  qActEnemy  gevinst når FIENDEN sin dronning er utløyst (ho kan takast)
#  king       kongen                   — bufferen: gjev alltid eit trekk
#  kFar       straff per felt kongen er frå slottet (H2)
#  blend      kor mykje materialet veg mot verdinettet i bladverdien
MATERIALS = {
    "sHome": 0.02,
    "sAct": 0.05,
    "qHome": 0.05,
    "qActBase": -0.08,
    "qActPerS": -0.035,
    "qActDen": -0.005,
    "qActNoKing": -0.12,
    "qActEnemy": 0.14,
    "king": 0.04,
    "kFar": -0.03,
    "blend": 0.30,
}

CASTLE = 22
KING_NEAR = 4          # Chebyshev-avstand frå slottet som framleis er «nær»


def material_sum(st, w: dict | None = None) -> float:
    """Materiell FORLEGELESGNAD frå `st.turn` si side: våre brikker +, deira −."""
    w = w or MATERIALS
    from . import engine as E
    from .features import _cheb
    me = st.turn
    s = 0.0
    my_soldiers = sum(1 for p in st.pieces if not p.captured and p.t == "s" and p.o == me)
    on_board = sum(1 for p in st.pieces if not p.captured)
    k = E.king(st)
    own_king = k is not None and not k.captured and k.o == me
    # Lars: «viss du ikkje kontrollerer kongen, er det nesten umogleg å verne
    # dronninga» — ANTEN manglande eigen konge, eller kongen langt frå slottet
    king_far = True
    if k is not None and not k.captured and k.o == me and _cheb(k.at, CASTLE) <= KING_NEAR:
        king_far = False
    for p in st.pieces:
        if p.captured:
            continue
        if p.o is None:          # nøytral konge (spelaren har ikkje teke han) tel ikkje
            continue
        sign = 1.0 if p.o == me else -1.0
        if p.t == "s":
            s += sign * (w["sAct"] if p.released else w["sHome"])
        elif p.t == "q":
            if not p.released:
                s += sign * w["qHome"]
            elif p.o == me:
                s += (w["qActBase"] + w["qActPerS"] * my_soldiers
                      + w["qActDen"] * on_board
                      + (w["qActNoKing"] if king_far else 0.0))
            else:
                s += w["qActEnemy"]
        elif p.t == "k":
            s += sign * (w["king"] + w["kFar"] * (min(_cheb(p.at, CASTLE), 14) / 14.0))
    return s


def blended_value(net_value: float, st, w: dict | None = None) -> float:
    """Bladverdi: net + blend·materiale, clampa til [-1, 1]."""
    w = w or MATERIALS
    v = net_value + w["blend"] * material_sum(st, w)
    return max(-1.0, min(1.0, v))


def parse_material_option(s: str) -> dict:
    """`setoption name Material value qActPerS=-0.05` -> dict (berre kjende nøklar)."""
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
