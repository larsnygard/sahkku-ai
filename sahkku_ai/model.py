"""AlphaZero-style network for sáhkku: conv trunk on the 3x15 board planes
+ global vector -> policy(8282, masked) + value(tanh).

Deliberately exportable: every op is conv/matmul/BN/ReLU so the identical
graph can run in plain JS (export.py emits weights + a JS runner).
"""
from __future__ import annotations
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_PLANES, N_GLOBAL, N_ACTIONS

BOARD_N = 45  # 3x15 flattened as a 3x15 image


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class Net(nn.Module):
    # Standardkonfigurasjonen som er trena og levert i engines/sahkku-nn.js:
    # 64 kanalar x 6 restblokker held seg på ~110 ms/trekk i rein JS (sjå
    # README-tabellen). Større nett er sterkare men treigare i nettlesaren.
    def __init__(self, ch: int = 64, blocks: int = 6, fc: int = 128):
        super().__init__()
        self.cfg = {"ch": ch, "blocks": blocks, "fc": fc}
        self.inp = nn.Conv2d(N_PLANES, ch, 3, padding=1, bias=False)
        self.inp_bn = nn.BatchNorm2d(ch)
        self.trunk = nn.ModuleList(ResBlock(ch) for _ in range(blocks))
        self.pol_conv = nn.Conv2d(ch, 8, 1)
        self.pol_bn = nn.BatchNorm2d(8)
        self.pol_fc = nn.Linear(8 * BOARD_N + N_GLOBAL, fc)
        self.pol_out = nn.Linear(fc, N_ACTIONS)
        self.val_conv = nn.Conv2d(ch, 4, 1)
        self.val_bn = nn.BatchNorm2d(4)
        self.val_fc1 = nn.Linear(4 * BOARD_N + N_GLOBAL, fc)
        self.val_fc2 = nn.Linear(fc, 1)

    def forward(self, planes, glob):
        # planes: (B, N_PLANES*45), glob: (B, N_GLOBAL)
        x = planes.view(-1, N_PLANES, 3, 15)
        x = F.relu(self.inp_bn(self.inp(x)))
        for blk in self.trunk:
            x = blk(x)
        p = F.relu(self.pol_bn(self.pol_conv(x))).flatten(1)
        p = F.relu(self.pol_fc(torch.cat([p, glob], 1)))
        policy = self.pol_out(p)
        v = F.relu(self.val_bn(self.val_conv(x))).flatten(1)
        v = F.relu(self.val_fc1(torch.cat([v, glob], 1)))
        value = torch.tanh(self.val_fc2(v)).squeeze(1)
        return policy, value

    def act(self, planes, glob, mask, temperature=0.0):
        """mask: (B, N_ACTIONS) 1/0 legal. Returns (action idx, value)."""
        pol, val = self(planes, glob)
        pol = pol.masked_fill(mask < 0.5, float("-inf"))
        if temperature <= 1e-6:
            a = pol.argmax(1)
        else:
            probs = torch.softmax(pol / temperature, 1)
            a = torch.multinomial(probs, 1).squeeze(1)
        return a, val

    def save(self, path, meta=None):
        torch.save({"state": self.state_dict(), "cfg": self.cfg,
                    "meta": meta or {}}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        net = cls(**ck["cfg"])
        net.load_state_dict(ck["state"])
        net.eval()
        net.meta = ck.get("meta", {})
        return net
