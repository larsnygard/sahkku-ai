"""Training loop: shards (SPG and/or self-play) -> policy+value net.

  * resume-friendly: --init loads weights from any checkpoint (meta carries
    net cfg; --init also accepts the output of a previous run)
  * masked policy cross-entropy (played-move soft targets) + MSE value
  * cosine LR schedule, AdamW, AMP off (P100 = cc6.0, fp16 conv is a mixed
    bag on 6.0; fp32 keeps parity with the JS export numerics)
  * periodic checkpoints checkpoints/<run>/step<N>.pt + latest.pt, with
    meta {step, samples, best_val, history} for fully resumable runs.

Usage: python -m sahkku_ai.train --shards 'runs/r1/*.npz data/*.npz' \
        --run r2 [--init checkpoints/r1/latest.pt] [--steps 2000] ...
"""
from __future__ import annotations
import argparse, glob as globmod, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from . import features as FT
from .model import Net
from .data import load_shards


def masked_policy_loss(logits, target, mask=None):
    """-Σ target·log p, i praksis over dei lovlege aksjonane.

    Målfordelinga (π) er alt bygd slik at ho er 0 utanfor den lovlege poolen
    (data.py og selfplay.py), så maska er implisitt; `mask` er teken med for
    å gjere kallstaden lesbar. Logp vert klipt på -30 mot -inf-søppel.
    """
    logp = F.log_softmax(logits, dim=1)
    return -(target * logp.clamp_min(-30)).sum(1).mean()


def train(run, shards_glob, init=None, steps=2000, batch=256, lr=2e-3,
          wd=1e-4, warmup=50, device="cuda", seed=13, val_holdout=0.05,
          ch=None, blocks=None, fc=None, log_every=20, save_every=200,
          extra=None, extra_weight=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    paths = sorted(set(sum([globmod.glob(g) for g in shards_glob.split()], [])))
    if not paths:
        raise SystemExit(f"ingen shard-filer i {shards_glob!r}")
    P, G, PI, V = load_shards(paths)
    n = len(V)
    idx = np.random.RandomState(seed).permutation(n)
    n_val = int(n * val_holdout)
    tr, va = idx[n_val:], idx[:n_val]
    print(f"treningsdata: {n} prøver frå {len(paths)} shards "
          f"({len(tr)} tren / {len(va)} val)")

    # Eit ekstra shard (t.d. menneskeparti) vert trekt inn i kvar minibatch med
    # fast vekt, so det tel sjølv om det er bittelite mot sjølvspelsdataen.
    ex_idx = None
    if extra and extra_weight > 0:
        n0 = len(V)
        P2, G2, PI2, V2 = load_shards([extra])
        P = np.concatenate([P, P2]); G = np.concatenate([G, G2])
        PI = np.concatenate([PI, PI2]); V = np.concatenate([V, V2])
        ex_idx = np.arange(n0, n0 + len(V2))
        print(f"ekstra shard: {len(V2)} prøver frå {extra} "
              f"(vekt {extra_weight:.2f} av kvar batch)")

    legal_rows = (PI > 0)
    if init:
        net = Net.load(init, map_location="cpu")
        print(f"init frå {init} (cfg {net.cfg}, meta {net.meta.get('steps','?')} steg)")
    else:
        net = Net(**{k: v for k, v in dict(ch=ch, blocks=blocks, fc=fc).items()
                     if v is not None})
        net.meta = {}
        print(f"random init cfg {net.cfg}")
    dev = torch.device(device)
    if dev.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(dev)} "
              f"(cc {torch.cuda.get_device_capability(dev)[0]}."
              f"{torch.cuda.get_device_capability(dev)[1]}, "
              f"torch {torch.__version__}, fp32)", flush=True)
    else:
        print(f"device: {dev} (torch {torch.__version__}, fp32)", flush=True)
    net.to(dev)

    P = torch.from_numpy(P); G = torch.from_numpy(G)
    PI = torch.from_numpy(PI); LM = torch.from_numpy(legal_rows.astype(np.float32))
    V = torch.from_numpy(V)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)

    def lr_at(step):
        if step < warmup:
            return lr * step / max(1, warmup)
        t = (step - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    ckdir = os.path.join("checkpoints", run)
    os.makedirs(ckdir, exist_ok=True)
    hist = net.meta.get("history", []) if init else []
    best_val = net.meta.get("best_val", float("inf")) if init else float("inf")
    t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        n_ex = int(round(batch * extra_weight)) if ex_idx is not None else 0
        rs = np.random.RandomState(seed + step)
        b = rs.choice(tr, batch - n_ex)
        if n_ex:
            b = np.concatenate([b, rs.choice(ex_idx, n_ex)])
        x = P[b].to(dev); gvec = G[b].to(dev)
        tgt = PI[b].to(dev); lm = LM[b].to(dev); v = V[b].to(dev)
        pol, val = net(x, gvec)
        loss_p = masked_policy_loss(pol, tgt, lm)
        loss_v = F.mse_loss(val, v)
        loss = loss_p + loss_v
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            el = time.time() - t0
            print(f"steg {step}/{steps}  tap {loss.item():.4f} "
                  f"(pol {loss_p.item():.4f} verd {loss_v.item():.4f})  "
                  f"lr {lr_at(step):.2e}  {el:.0f}s", flush=True)
        if step % save_every == 0 or step == steps:
            vp, vv = evaluate(net, dev, P, G, PI, V, va, batch=1024)
            hist.append({"step": step, "val_policy_xent": vp, "val_value_mse": vv})
            print(f"  VAL: policy-xent {vp:.4f}  verd-MSE {vv:.4f}", flush=True)
            net.meta = {"run": run, "steps": step, "samples": n,
                        "best_val": min(best_val, vp), "history": hist,
                        "seed": seed}
            net.save(os.path.join(ckdir, f"step{step}.pt"), net.meta)
            net.save(os.path.join(ckdir, "latest.pt"), net.meta)
            if vp < best_val:
                best_val = vp
                net.save(os.path.join(ckdir, "best.pt"), net.meta)
    print(f"trening ferdig på {time.time()-t0:.0f}s → checkpoints/{run}/latest.pt")
    return os.path.join(ckdir, "latest.pt")


@torch.no_grad()
def evaluate(net, dev, P, G, PI, V, va, batch=1024):
    tot_p, tot_v, cnt = 0.0, 0.0, 0
    for i in range(0, len(va), batch):
        b = va[i:i + batch]
        x = P[b].to(dev); g = G[b].to(dev)
        tgt = PI[b].to(dev); v = V[b].to(dev)
        pol, val = net(x, g)
        lm = (tgt > 0).float()
        tot_p += masked_policy_loss(pol, tgt, lm).item() * len(b)
        tot_v += F.mse_loss(val, v, reduction="sum").item()
        cnt += len(b)
    return tot_p / cnt, tot_v / cnt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--shards", required=True,
                    help="space-separated globs, e.g. 'runs/r1/*.npz data/*.npz'")
    ap.add_argument("--init", default=None, help="checkpoint .pt to resume from")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ch", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--fc", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=200)
    a = ap.parse_args()
    train(a.run, a.shards, init=a.init, steps=a.steps, batch=a.batch, lr=a.lr,
          wd=a.wd, warmup=a.warmup, device=a.device, seed=a.seed,
          ch=a.ch, blocks=a.blocks, fc=a.fc,
          log_every=a.log_every, save_every=a.save_every)
