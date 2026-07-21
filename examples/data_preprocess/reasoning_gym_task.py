"""Generate train/val parquets for a reasoning-gym task (e.g. chain_sum) in the
format expected by verl + the multi_thread (math-verify) reward manager.

The multi_thread reward manager scores by math-verifying the completion against
\\boxed{ground_truth}, so this is only suitable for tasks whose answer is a
number or math expression (chain_sum, gcd, basic_arithmetic, gsm_symbolic, ...).

Usage:
    python examples/data_preprocess/reasoning_gym_task.py --task chain_sum \
        --local_dir ~/data/reasoning_gym
"""

import argparse
import os

import pandas as pd

import reasoning_gym


def build(task, split, size, seed):
    dataset = reasoning_gym.create_dataset(task, size=size, seed=seed)
    rows = []
    for idx, entry in enumerate(dataset):
        rows.append(
            {
                "data_source": f"reasoning_gym/{task}",
                "prompt": [
                    {
                        "role": "user",
                        "content": entry["question"],
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(entry["answer"])},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": entry["question"],
                },
            }
        )
        if idx == 0:
            print(f"[{split}] example question: {entry['question']!r}")
            print(f"[{split}] example ground_truth: {entry['answer']!r}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="chain_sum")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=1042)
    parser.add_argument("--local_dir", default="~/data/reasoning_gym")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.expanduser(args.local_dir), args.task)
    os.makedirs(out_dir, exist_ok=True)

    train_df = build(args.task, "train", args.train_size, args.train_seed)
    val_df = build(args.task, "val", args.val_size, args.val_seed)

    train_df.to_parquet(os.path.join(out_dir, "train.parquet"))
    val_df.to_parquet(os.path.join(out_dir, "val.parquet"))
    print(f"wrote {len(train_df)} train / {len(val_df)} val rows to {out_dir}")
