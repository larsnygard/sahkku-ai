"""Utvid eit checkpoint med fleire globale trekk, utan å endre funksjonen.

Når N_GLOBAL veks (t.d. 10 -> 15 med konge- og terningtrekka), passar ikkje
dei gamle vektene til inngangsvektoren lenger. Men dei globale tala ligg
SIST i inngangsvektoren til pol_fc og val_fc1, so vi kan leggje til nye
kolonnar sett til 0: då multipliserer dei nye trekka med 0 og nettet reknar
NØYAKTIG som før. Treninga kan so lære seg å bruke dei nye trekka.

Køyr:  .venv/bin/python tools/expand_net.py <inn.pt> <ut.pt>
"""
import sys
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from sahkku_ai.features import N_GLOBAL
from sahkku_ai.model import Net

# dei to laga som tek inn globale tal: (namn, breidd av conv-delen)
WIDE = [("pol_fc", 8 * 45), ("val_fc1", 4 * 45)]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    old = ck["state"]

    net = Net(**cfg)
    new = net.state_dict()
    copied, widened = [], []

    for k, v in new.items():
        src_t = old.get(k)
        if src_t is None:
            raise SystemExit(f"manglar {k} i {src}")
        if src_t.shape == v.shape:
            v.copy_(src_t)
            continue
        # utvida lag: kopier dei gamle kolonnane, lat dei nye stå på 0
        if k.endswith(".weight") and v.dim() == 2 and v.shape[0] == src_t.shape[0]:
            name = k.split(".")[0]
            base = dict(WIDE).get(name)
            if base is None:
                raise SystemExit(f"unknown expansion on {k}: {tuple(src_t.shape)} -> {tuple(v.shape)}")
            n_old = src_t.shape[1] - (src_t.shape[1] - base - (src_t.shape[1] - base))
            # gamle kolonnar = base conv-kolonnar + (N_GLOBAL - nye) globale
            keep = base + (src_t.shape[1] - base)
            v.zero_()
            v[:, :src_t.shape[1]] = src_t          # alle gamle kolonnar fyrst
            widened.append((k, tuple(src_t.shape), tuple(v.shape),
                            N_GLOBAL - (src_t.shape[1] - base)))
        elif k.endswith(".bias") or "bias" in k or "running_" in k or "num_batches" in k:
            v.copy_(src_t)
        else:
            raise SystemExit(f"cannot expand {k}: {tuple(src_t.shape)} -> {tuple(v.shape)}")

    net.load_state_dict(new)
    meta = ck.get("meta", {})
    meta["expanded_from"] = src
    meta["globals_added"] = N_GLOBAL - (old["pol_fc.weight"].shape[1] - 8 * 45)
    net.save(dst, meta)

    print(f"wrote {dst}")
    print(f"  globale tal: {old['pol_fc.weight'].shape[1] - 8 * 45} -> {N_GLOBAL}")
    for k, a, b, n in widened:
        print(f"  {k}: {a} -> {b} ({n} nye kolonnar sett til 0)")


if __name__ == "__main__":
    main()
