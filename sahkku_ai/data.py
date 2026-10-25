"""Dataset pipelines.

1. SPG collections -> supervised samples. Each played move becomes one
   sample: (planes, globals, policy target over legal ids, outcome value
   from side-to-move's perspective). Replay uses spg.game_from_spg on the
   parity-verified engine, so SPG legality == engine legality; games that
   diverge are skipped (strict raise) — reported by dataset_from_spg.
2. Self-play samples written by selfplay.py (same shard format).

Shard format (.npz, compressed):
  planes (N, 16*45) f32 · glob (N, 10) f32 · policy (N, 8282) f32
  value (N,) f32 (in [-1,1], side-to-move POV)
Policy target: one-hot 1.0 on the played move (self-play: Dirichlet-free,
the P mover *is* the target); for SPG games a soft version: 0.9 played
move + 0.1 spread over the rest of the legal ids (human CDP moves keep a
small plateau the net can fit).
"""
from __future__ import annotations
import glob as globmod
import numpy as np
from . import engine as E
from . import features as FT
from .spg import parse_spg, game_from_spg

N_PLANES_FLAT = FT.N_PLANES * 45
N_GLOBAL = FT.N_GLOBAL
N_ACTIONS = FT.N_ACTIONS


def _encode_sample(st, mv_obj, played_weight=0.9):
    """(planes, glob, policy, value_pov) for a state + the move played in it."""
    legal = [a for a, _ in FT.legal_ids(st)]
    aid = FT.move_id(st, mv_obj)
    pol = np.zeros(N_ACTIONS, np.float32)
    if aid not in legal:
        return None
    rest = (1.0 - played_weight) / max(1, len(legal) - 1)
    for a in legal:
        pol[a] = rest
    pol[aid] = played_weight
    planes, g = FT.encode_state(st)
    return planes.reshape(-1), g, pol


def spg_samples(text: str):
    """Yield samples from one SPG game text (skip whole game on divergence)."""
    p = parse_spg(text)
    result = p.get("result") or "*"
    val_menn = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
    try:
        _final, samples = game_from_spg(text, strict=True)
    except ValueError:
        return
    for st_before, mv in samples:
        s = _encode_sample(st_before, mv, played_weight=0.9)
        if s is None:
            continue
        planes, g, pol = s
        v = val_menn if st_before.turn == 0 else -val_menn
        yield planes, g, pol, np.float32(v)


def load_spg_dir(path: str, verbose=True):
    files = sorted(set(globmod.glob(path + "/*.spg") +
                       globmod.glob(path + "/**/*.spg", recursive=True)))
    out, skipped = [], 0
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        n0 = len(out)
        out.extend(spg_samples(txt))
        if len(out) == n0 and verbose:
            skipped += 1
            print(f"  hoppa over (skilde/ugyldig): {fp}")
    if verbose:
        print(f"SPG-filer: {len(files)} fila, {skipped} hoppa over, {len(out)} prøver")
    return out


def save_shard(samples, path: str):
    planes = np.stack([s[0] for s in samples])
    glob_ = np.stack([s[1] for s in samples])
    pol = np.stack([s[2] for s in samples])
    val = np.stack([s[3] for s in samples])
    np.savez_compressed(path, planes=planes, glob=glob_, policy=pol, value=val)


def load_shards(paths, limit=0):
    P, G, PI, V = [], [], [], []
    n = 0
    for p in paths:
        z = np.load(p)
        P.append(z["planes"]); G.append(z["glob"])
        PI.append(z["policy"]); V.append(z["value"])
        n += len(z["value"])
        if limit and n >= limit:
            break
    return (np.concatenate(P), np.concatenate(G),
            np.concatenate(PI), np.concatenate(V))


def dataset_from_spg(path: str, out: str):
    s = load_spg_dir(path)
    if not s:
        print("INGA PRØVE — tomt SPG-utgangsmateriell")
        return 0
    save_shard(s, out)
    print(f"SPG -> {out}: {len(s)} prøver")
    return len(s)
