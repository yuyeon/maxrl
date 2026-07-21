"""Generate and score a completion for every row of one or more verl parquets.

Run after training (against the merged HF checkpoint) to get per-example
completions + scores over the full train and val sets, using the same
math-verify scorer as the multi_thread reward manager.

Usage:
    python smollm/eval_parquet.py \
        --model /path/to/merged_hf_model \
        --parquets ~/data/reasoning_gym/chain_sum/train.parquet \
                   ~/data/reasoning_gym/chain_sum/val.parquet \
        --out eval_results/chain_sum_step300

Outputs:
    <out>/samples.jsonl  one line per input row: question, ground truth,
                         completions with scores
    <out>/summary.json   per-parquet accuracy and pass@n
"""

import argparse
import json
import os

import pandas as pd
from vllm import LLM, SamplingParams

from verl.utils.reward_score.math_verify import compute_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="merged HF checkpoint dir (from scripts/model_merger.py) or hub id")
    parser.add_argument("--parquets", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=1,
                        help="completions per prompt (use temperature > 0 if > 1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy capability read; 1.0 matches training rollouts")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.samples > 1 and args.temperature == 0.0:
        parser.error("--samples > 1 with --temperature 0.0 produces identical samples")

    frames = []
    for path in args.parquets:
        df = pd.read_parquet(path)
        df["_source"] = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(data)} rows from {len(frames)} parquet(s)")

    # parquet round-trips the chat prompt as an array of {role, content} dicts
    conversations = [[dict(m) for m in row] for row in data["prompt"]]

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        n=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.chat(conversations, sampling)

    os.makedirs(args.out, exist_ok=True)
    per_source = {}
    with open(os.path.join(args.out, "samples.jsonl"), "w") as f:
        for (_, row), output in zip(data.iterrows(), outputs):
            ground_truth = str(row["reward_model"]["ground_truth"])
            completions = []
            for sample in output.outputs:
                score = float(compute_score(sample.text, ground_truth))
                completions.append({"text": sample.text, "score": score})
            scores = [c["score"] for c in completions]
            record = {
                "source": row["_source"],
                "index": int(row["extra_info"]["index"]),
                "question": row["prompt"][0]["content"],
                "ground_truth": ground_truth,
                "completions": completions,
                "mean_score": sum(scores) / len(scores),
                "any_correct": max(scores) >= 1.0,
            }
            f.write(json.dumps(record) + "\n")
            stats = per_source.setdefault(row["_source"], {"n": 0, "mean_score": 0.0, "pass_at_n": 0})
            stats["n"] += 1
            stats["mean_score"] += record["mean_score"]
            stats["pass_at_n"] += int(record["any_correct"])

    summary = {
        "model": args.model,
        "samples_per_prompt": args.samples,
        "temperature": args.temperature,
        "per_source": {
            src: {
                "n": s["n"],
                "mean_score": s["mean_score"] / s["n"],
                f"pass_at_{args.samples}": s["pass_at_n"] / s["n"],
            }
            for src, s in per_source.items()
        },
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["per_source"], indent=2))
    print(f"wrote {args.out}/samples.jsonl and summary.json")


if __name__ == "__main__":
    main()


