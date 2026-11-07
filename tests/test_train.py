"""Focused tests for training mode, validation, and small datasets."""
from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np
import torch

from sahkku_ai import features as FT
from sahkku_ai.model import Net
from sahkku_ai.train import evaluate, train


def samples(count=4):
    rng = np.random.default_rng(7)
    planes = rng.normal(size=(count, FT.N_PLANES * 45)).astype(np.float32)
    glob = rng.normal(size=(count, FT.N_GLOBAL)).astype(np.float32)
    policy = np.zeros((count, FT.N_ACTIONS), dtype=np.float32)
    policy[:, 0] = 1.0
    value = np.linspace(-1, 1, count, dtype=np.float32)
    return planes, glob, policy, value


def main():
    planes, glob, policy, value = samples()
    net = Net(ch=2, blocks=0, fc=4)
    net.train()
    before = net.inp_bn.running_mean.clone()
    evaluate(net, torch.device("cpu"), torch.from_numpy(planes),
             torch.from_numpy(glob), torch.from_numpy(policy),
             torch.from_numpy(value), np.array([0, 1]))
    assert net.training, "evaluation must restore the previous training mode"
    assert torch.equal(before, net.inp_bn.running_mean), "validation must not update BatchNorm"

    with tempfile.TemporaryDirectory() as tmp:
        shard = pathlib.Path(tmp) / "samples.npz"
        np.savez(shard, planes=planes, glob=glob, policy=policy, value=value)
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)
            first = train("first", str(shard), steps=1, batch=2, warmup=0,
                          device="cpu", ch=2, blocks=0, fc=4,
                          log_every=1, save_every=1)
            first_ck = torch.load(first, map_location="cpu", weights_only=False)
            tracked = first_ck["state"]["inp_bn.num_batches_tracked"].item()
            assert tracked == 1, f"expected one training batch, got {tracked}"

            second = train("second", str(shard), init=first, steps=1, batch=2,
                           warmup=0, device="cpu", log_every=1, save_every=1)
            second_ck = torch.load(second, map_location="cpu", weights_only=False)
            resumed_tracked = second_ck["state"]["inp_bn.num_batches_tracked"].item()
            assert resumed_tracked == tracked + 1, (
                "training from a checkpoint must update BatchNorm statistics")
        finally:
            os.chdir(old_cwd)

    with tempfile.TemporaryDirectory() as tmp:
        shard = pathlib.Path(tmp) / "one.npz"
        one = samples(1)
        np.savez(shard, planes=one[0], glob=one[1], policy=one[2], value=one[3])
        try:
            train("too_small", str(shard), steps=1, batch=1, device="cpu")
        except ValueError as exc:
            assert "at least 2 samples" in str(exc)
        else:
            raise AssertionError("a one-sample dataset must be rejected")

    print("test_train: pass")


if __name__ == "__main__":
    main()
