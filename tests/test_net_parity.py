"""Net/feature parity: PyTorch (fp32, CPU) vs the exported pure-JS runner.

Feeds the SAME FEN states through both:
  * planes + globals must be elementwise identical (1e-4)
  * legal action id sets identical
  * policy logits at legal ids + value: |Δ| < 1e-2 (fp32 conv order noise)
"""
import argparse, json, os, subprocess, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sahkku_ai import engine as E
from sahkku_ai import features as FT
from sahkku_ai.fen import from_fen, to_fen
from sahkku_ai.model import Net
import random


def exported_source():
    """Checkpointet som engines/sahkku-nn.js vart eksportert frå.

    Vektene har `meta.source` (stien eksporten fekk). Testen må samanlikne
    mot NETTOPP det checkpointet — elles kan han plukke eit nyare .pt som
    ikkje ligg i .js-fila og melde falsk avvik.
    """
    eng = os.path.join(os.path.dirname(__file__), "..", "engines")
    # 1) eksporten legg att si eiga kjelde: engines/sahkku-nn.pt
    copy = os.path.join(eng, "sahkku-nn.pt")
    if os.path.exists(copy):
        return copy
    # 2) elles stien eksporten hugsar (same maskin, same arbeidskopi)
    wpath = os.path.join(eng, "sahkku-nn.weights.json")
    try:
        with open(wpath) as f:
            src = json.load(f).get("meta", {}).get("source")
    except OSError:
        src = None
    return src if src and os.path.exists(src) else None


def newest_checkpoint():
    """Nyaste .pt under checkpoints/ (best.pt/latest.pt), elles ein tydeleg feil."""
    import glob as g
    cands = [p for p in g.glob("checkpoints/**/*.pt", recursive=True)
             if p.endswith(("latest.pt", "best.pt", "init.pt"))]
    if not cands:
        raise SystemExit(
            "fann ingen checkpoint under checkpoints/ — lag ein først, t.d.\n"
            "  .venv/bin/python -m sahkku_ai.train --run smoke --shards 'data/*.npz' --steps 30")
    return max(cands, key=os.path.getmtime)


probe_path = None


def main(net_path=None, n=25, js_path=None):
    net_path = net_path or exported_source() or newest_checkpoint()
    print(f"nett: {net_path}")
    if js_path:
        global probe_path
        probe_path = js_path
        print(f"js:   {js_path}")
    net = Net.load(net_path, map_location="cpu")
    torch.set_grad_enabled(True)
    rnd = random.Random(11)
    fens = []
    # diverse random positions via engine play
    for g in range(8):
        st = E.State(first=rnd.randint(0, 1))
        E.opening_roll(st, rnd.random)
        for _ in range(80):
            if st.winner is not None:
                break
            if st.phase == "roll":
                E.roll(st, rnd.random); st.phase = "move"
            lm = E.legal_moves(st, st.turn)
            if not lm:
                if E.can_reroll_x(st):
                    E.reroll_x(st, rnd.random); st.phase = "move"; continue
                E.pass_remaining(st); continue
            fens.append(to_fen(st))
            mv = lm[rnd.randrange(len(lm))]
            E.apply_move(st, mv); E.settle(st)
    fens = fens[:n]

    py = []
    for fen in fens:
        st = from_fen(fen)
        planes, g = FT.encode_state(st)
        ids = [a for a, _ in FT.legal_ids(st)]
        x = torch.from_numpy(planes.reshape(-1)[None, :])
        gg = torch.from_numpy(g[None, :])
        pol, val = net(x, gg)
        py.append({"fen": fen, "planes": planes.reshape(-1), "glob": g,
                   "legal": ids,
                   "policy": pol[0].detach().numpy()[ids],
                   "value": val[0].item()})

    probe = probe_path or os.path.join(os.path.dirname(__file__), "..", "engines", "sahkku-nn.js")
    payload = json.dumps([{"fen": f} for f in fens])
    r = subprocess.run(["node", probe, "features-test"], input=payload.encode(),
                       capture_output=True, timeout=120)
    if r.returncode != 0:
        print("JS FEILA:", r.stderr.decode()[:500]); sys.exit(1)
    js = json.loads(r.stdout.decode())

    fails = 0
    for a, b in zip(py, js):
        pa = np.asarray(b["planes"], np.float32)
        ga = np.asarray(b["glob"], np.float32)
        dP = np.abs(a["planes"] - pa).max()
        dG = np.abs(a["glob"] - ga).max()
        if dP > 1e-4 or dG > 1e-4:
            print(f"FAIL features {a['fen']}: Δplanes {dP:.2e} Δglob {dG:.2e}")
            fails += 1
        if sorted(a["legal"]) != sorted(b["legal"]):
            print(f"FAIL legal {a['fen']}: py {a['legal']} js {b['legal']}")
            fails += 1
            continue
        pol_js = np.asarray(b["policy"], np.float32)
        dPol = np.abs(a["policy"] - pol_js).max()
        dVal = abs(a["value"] - b["value"])
        if dPol > 1e-2:
            print(f"FAIL policy {a['fen']}: Δ {dPol:.2e}")
            fails += 1
        if dVal > 1e-2:
            print(f"FAIL value {a['fen']}: py {a['value']:.4f} js {b['value']:.4f}")
            fails += 1
    if fails:
        print(f"NETT-PARITET FEILA: {fails}/{len(py)} posisjonar")
        sys.exit(1)
    print(f"NETT-PARITET OK: {len(py)} posisjonar — plan, legal, policy, value like i PyTorch og JS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("net", nargs="?", default=None,
                    help="checkpoint .pt (standard: nyaste under checkpoints/)")
    ap.add_argument("-n", type=int, default=25, help="talet på posisjonar")
    ap.add_argument("--js", default=None,
                    help="den eksporterte .js-fila som høyrer til .pt-en (standard: engines/sahkku-nn.js)")
    a = ap.parse_args()
    main(a.net, a.n, a.js)
