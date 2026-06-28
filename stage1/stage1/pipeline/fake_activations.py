#!/usr/bin/env python
"""Write synthetic activation caches for offline pipeline testing (no GPU)."""

import argparse

import numpy as np

from stage1.common.config import load_defaults, load_split
from stage1.common.paths import ACTIVATIONS_DIR, data_file
from stage1.icrl.boundaries import get_first_post_discovery_turn
from stage1.icrl.mock_data import write_mock_icrl
from stage1.icrl.schema import load_conversations


def write_fake_activations(
    icrl_path,
    *,
    n_layers: int = 36,
    hidden_dim: int = 4096,
    seed: int = 0,
):
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    split = load_split()
    rng = np.random.default_rng(seed)

    direction = rng.standard_normal((n_layers, hidden_dim)).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    for conv in load_conversations(icrl_path):
        post_turn = get_first_post_discovery_turn(conv)
        if post_turn is None:
            continue

        seq_len = 128
        acts = rng.standard_normal((n_layers, seq_len, hidden_dim)).astype(np.float16)
        labels = np.full(seq_len, -1, dtype=np.int8)

        pre_idx = list(range(20, 40))
        post_idx = list(range(40, 60))
        for i in pre_idx:
            labels[i] = 0
        for i in post_idx:
            labels[i] = 1

        offset = -1.5 if conv.criterion_id in split["train"] else 1.5
        for layer in range(n_layers):
            for i in pre_idx:
                acts[layer, i] += (offset * direction[layer]).astype(np.float16)
            for i in post_idx:
                acts[layer, i] += (-offset * direction[layer]).astype(np.float16)

        out = ACTIVATIONS_DIR / f"{conv.conv_id}.npz"
        np.savez_compressed(
            out,
            layer_activations=acts,
            token_labels=labels,
            pre_indices=np.array(pre_idx, dtype=np.int32),
            post_indices=np.array(post_idx, dtype=np.int32),
            criterion_id=conv.criterion_id,
            conv_id=conv.conv_id,
        )
        print(f"  fake activations -> {out}", flush=True)


def main():
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--icrl", default=data_file("mock_icrl.json"))
    ap.add_argument("--regenerate-mock", action="store_true")
    args = ap.parse_args()

    if args.regenerate_mock:
        write_mock_icrl(args.icrl)

    write_fake_activations(
        args.icrl,
        n_layers=defaults["n_layers"],
        hidden_dim=defaults["hidden_dim"],
    )


if __name__ == "__main__":
    main()
