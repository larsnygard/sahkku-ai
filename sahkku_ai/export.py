"""Export a trained checkpoint into the S-UCI JS engine.

  * weights JSON (float32 little-endian base64, per-tensor) — the identical
    graph as model.py, executed by the pure-JS runner (engines/sahkku-nn.js
    template in SnartSahkku repo, copied to this repo's engines/).
  * engines/sahkku-nn.js — self-contained S-UCI engine (vendored rules +
    this weights file), zero npm deps, runs on Node 18+ and in browsers.

Usage: python -m sahkku_ai.export --net checkpoints/r1/best.pt \
        --out engines/sahkku-nn --quant fp32
"""
from __future__ import annotations
import argparse, base64, io, json, os, shutil, sys
import numpy as np
import torch

from .model import Net
from .features import N_PLANES, N_GLOBAL, N_ACTIONS


def tensor_b64(t):
    a = t.detach().cpu().numpy().astype(np.float32)
    return base64.b64encode(a.tobytes()).decode("ascii")


def export(net_path, out_prefix, quant="fp32", name="SahkkuNN 1.0"):
    net = Net.load(net_path, map_location="cpu")
    sd = net.state_dict()
    blobs, shapes = {}, {}
    for k, v in sd.items():
        blobs[k] = tensor_b64(v)
        shapes[k] = list(v.shape)
    # Checkpointet følgjer med eksporten (engines/<namn>.pt), slik at vekter,
    # JS-kode og kjeldenett alltid er same nett — også i ein fersk klone, der
    # stien i `source` ikkje finst.
    cp_path = out_prefix + ".pt"
    try:
        shutil.copyfile(net_path, cp_path)
    except (shutil.SameFileError, FileNotFoundError):
        cp_path = net_path
    meta = {"cfg": net.cfg, "checkpoint_meta": net.meta,
            "planes": N_PLANES, "glob": N_GLOBAL, "actions": N_ACTIONS,
            "quant": quant, "source": net_path,
            "checkpoint": os.path.basename(cp_path)}
    wpath = out_prefix + ".weights.json"
    with open(wpath, "w") as f:
        json.dump({"meta": meta, "shapes": shapes, "tensors": blobs}, f)
    # runner js: template + INJECTED canonical rules (js/engine.js, js/fen.js)
    root = os.path.join(os.path.dirname(__file__), "..")
    tpl = os.path.join(root, "engines", "runner_template.js")
    with open(tpl, "r", encoding="utf-8") as f:
        src = f.read()
    rules = open(os.path.join(root, "vendor", "engine.js"), encoding="utf-8").read()
    fen = open(os.path.join(root, "vendor", "fen.js"), encoding="utf-8").read()
    src = src.replace("/*§RULES§*/", rules).replace("/*§FEN§*/", fen)
    src = src.replace("__N_GLOBAL__", str(N_GLOBAL))
    src = src.replace("__WEIGHTS_FILE__", os.path.basename(wpath))
    src = src.replace("__ENGINE_NAME__", name)
    rpath = out_prefix + ".js"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"export: {rpath} + {wpath} + {cp_path}")
    print(f"try:   node {rpath}  (S-UCI: uci -> uciok, position + go)")
    return rpath, wpath


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--out", default="engines/sahkku-nn")
    ap.add_argument("--quant", default="fp32")
    ap.add_argument("--name", default="SahkkuNN 1.0",
                    help="S-UCI-id, t.d. «AIL0.1»")
    a = ap.parse_args()
    export(a.net, a.out, a.quant, a.name)
