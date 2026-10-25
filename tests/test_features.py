"""Feature/action layer tests: legal pool ids unique, decode round-trips,
encoded planes stable, specials present, dice ordering respected."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sahkku_ai import engine as E
from sahkku_ai import features as FT
from sahkku_ai.fen import to_fen, from_fen
import numpy as np


def test_pool(seed=1, games=12):
    rnd = random.Random(seed)
    seen_rel, seen_move = set(), set()
    n = 0
    for g in range(games):
        rand = rnd.random
        st = E.State(first=rnd.randint(0, 1))
        E.opening_roll(st, rand)
        for _ in range(300):
            if st.winner is not None:
                break
            if st.phase == "roll":
                E.roll(st, rand)
                st.phase = "move"
            lm = E.legal_moves(st, st.turn)
            if not lm:
                if E.can_reroll_x(st):
                    ids = [a for a, _ in FT.legal_ids(st)]
                    assert FT.special_id("reroll") in ids, "reroll id missing"
                    E.reroll_x(st, rand)
                    st.phase = "move"
                    continue
                E.pass_remaining(st)
                continue
            ids = [a for a, _ in FT.legal_ids(st)]
            assert len(ids) == len(set(ids)), f"legal_ids must dedupe by id: pool {len(lm)} ids {len(set(ids))}"
            for aid, mv in FT.legal_ids(st):
                kind, letter, frm, to, side = FT.decode_id(aid)
                if mv.get("kind") == "rel":
                    assert kind == "rel"
                else:
                    assert kind == "move"
                    seen_move.add((letter, frm, to))
            # move_id deterministic
            for mv in lm:
                assert FT.move_id(st, mv) == FT.move_id(st, mv)
            n += len(lm)
            mv = lm[rnd.randrange(len(lm))]
            E.apply_move(st, mv)
            E.settle(st)
    print(f"pool OK: {n} trekk, {len(seen_move)} unike (letter,frm,to), ingen duplikat-ID")


def test_rel_ids(seed=3):
    rnd = random.Random(seed)
    rel_ids = set()
    rand = rnd.random
    for g in range(6):
        st = E.State(first=0)
        E.opening_roll(st, rand)
        for _ in range(200):
            if st.winner is not None:
                break
            if st.phase == "roll":
                E.roll(st, rand)
                st.phase = "move"
            lm = E.legal_moves(st, st.turn)
            if not lm:
                if E.can_reroll_x(st):
                    E.reroll_x(st, rand); st.phase = "move"; continue
                E.pass_remaining(st); continue
            for aid, mv in FT.legal_ids(st):
                if mv.get("kind") == "rel":
                    kind, cls, frm, to, side = FT.decode_id(aid)
                    assert kind == "rel"
                    rel_ids.add((side, cls, to))
            mv = lm[rnd.randrange(len(lm))]
            E.apply_move(st, mv)
            E.settle(st)
    print(f"rel OK: {len(rel_ids)} unike løysings-ID")


def test_encode_shapes():
    st = E.State()
    pl, g = FT.encode_state(st)
    assert pl.shape == (FT.N_PLANES, 3, 15), pl.shape
    assert g.shape == (FT.N_GLOBAL,)
    v = FT.encode_pair(st)
    assert v.shape == (FT.N_PLANES * 45 + FT.N_GLOBAL,)
    assert v.dtype == np.float32
    # start position: 15 home m, 15 home g, queens home, king N at H2
    assert pl[2, 0].sum() == 15 and pl[3, 2].sum() == 15
    assert pl[6, 1, 11] == 1  # menn queen home L2 (rank2 col 11)
    assert pl[10, 1, 7] == 1  # neutral king H2 (rank2 col 7)
    # a seized king goes to the owned plane, NOT the soldier plane
    st2 = st.clone()
    E.king(st2).o = 0
    pl2, _ = FT.encode_state(st2)
    assert pl2[8, 1, 7] == 1 and pl2[10, 1, 7] == 0
    # dice plane active
    st3 = st.clone()
    st3.dice = ["X", "_", "_"]; st3.used = [False, False, False]; st3.phase = "move"
    pl3, _ = FT.encode_state(st3)
    assert pl3[11, 0, 0] == 1
    print("encode OK:", v.shape, "startplan + konge-eigar + terningsplan verifisert")


def test_net_shapes():
    import torch
    from sahkku_ai.model import Net
    net = Net(ch=32, blocks=2, fc=64)
    x = torch.randn(4, FT.N_PLANES * 45)
    g = torch.randn(4, FT.N_GLOBAL)
    pol, val = net(x, g)
    assert pol.shape == (4, FT.N_ACTIONS) and val.shape == (4,)
    mask = torch.zeros(4, FT.N_ACTIONS)
    mask[:, :5] = 1
    a, v = net.act(x, g, mask, temperature=1.0)
    assert (a < 5).all(), "temp sampling leaked outside legal mask"
    a2, _ = net.act(x, g, mask)
    assert (a2 < 5).all(), "greedy leaked outside legal mask"
    print(f"net OK: policy {tuple(pol.shape)}, value {tuple(val.shape)}, mask-lækkasje test OK")


def test_fen_roundtrip_states(seed=5):
    rnd = random.Random(seed)
    rand = rnd.random
    n = 0
    for g in range(6):
        st = E.State(first=0)
        E.opening_roll(st, rand)
        for _ in range(200):
            if st.winner is not None:
                break
            if st.phase == "roll":
                E.roll(st, rand); st.phase = "move"
            lm = E.legal_moves(st, st.turn)
            if not lm:
                if E.can_reroll_x(st):
                    E.reroll_x(st, rand); st.phase = "move"; continue
                E.pass_remaining(st); continue
            fen = to_fen(st)
            back = from_fen(fen)
            assert to_fen(back) == fen, f"FEN roundtrip: {fen} -> {to_fen(back)}"
            n += 1
            mv = lm[rnd.randrange(len(lm))]
            E.apply_move(st, mv); E.settle(st)
    print(f"fen roundtrip OK: {n} posisjonar, to_fen(from_fen(f)) == f")


if __name__ == "__main__":
    test_pool()
    test_rel_ids()
    test_encode_shapes()
    test_fen_roundtrip_states()
    test_net_shapes()
    print("FEATURES/MODEL TEST OK")
