#!/usr/bin/env python3
"""Sweep a model across reasoning-gym tasks to find trainable ones.

For each task, samples N completions per prompt and reports mean score,
pass@1, pass@k, and the fraction of prompts in the "trainable band"
(0 < pass rate < 1) where group-relative RL estimators (maxrl/grpo/rloo)
receive gradient signal.

Requires `pip install reasoning-gym`, plus one generation backend:
vllm (GPU sweep) or transformers+torch (small local smoke test).

Examples:
    # list all registered reasoning-gym tasks
    python smollm/rg_sweep.py --list

    # small local smoke test on CPU/MPS
    python smollm/rg_sweep.py --backend hf --tasks leg_counting,chain_sum \
        --size 10 --samples 4

    # full sweep on a GPU machine
    python smollm/rg_sweep.py --backend vllm --all --size 50 --samples 32
"""

import argparse
import json
import os
import re
import time

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Starter set spanning categories, biased toward tasks a small model has a
# chance on. Use --all (or --tasks) to override.
DEFAULT_TASKS = [
    "basic_arithmetic",
    "chain_sum",
    "leg_counting",
    "gsm_symbolic",
    "simple_equations",
    "number_sorting",
    "letter_counting",
    "spell_backward",
    "syllogism",
    "propositional_logic",
    "family_relationships",
    "maze",
    "mini_sudoku",
    "word_ladder",
]

# Mirrors the reasoning-gym convention: final answer inside <answer> tags.
# The one-shot example matters: for SmolLM2-360M it lifts tag compliance from
# ~3% to ~50% and pass@1 on easy chain_sum from 0.00 to 0.44.
SYSTEM_PROMPT = (
    "You are a helpful assistant that solves reasoning problems. Think step by "
    "step, then end your reply with the final answer between <answer> and "
    "</answer> tags.\n\n"
    "Example:\nQuestion: State the final answer to the following arithmetic "
    "problem: 2 + 5 =\nYour reply: 2 + 5 = 7. <answer>7</answer>"
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(completion: str):
    """Return the last <answer>...</answer> span, or None if absent."""
    matches = ANSWER_RE.findall(completion)
    return matches[-1].strip() if matches else None


def build_conversations(entries):
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": entry["question"]},
        ]
        for entry in entries
    ]


def generate_vllm(args):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        n=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    def generate(convs):
        outputs = llm.chat(convs, sampling_params)
        return [[o.text for o in out.outputs] for out in outputs]

    return generate


def generate_hf(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    model.to(device)
    model.eval()

    def generate(convs):
        results = []
        for conv in convs:
            inputs = tokenizer.apply_chat_template(
                conv, add_generation_prompt=True, return_tensors="pt",
                return_dict=True,
            ).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.samples,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completions = tokenizer.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            results.append(completions)
        return results

    return generate


def score_task(dataset, entries, completions_per_prompt, pass_threshold):
    """Score all completions; returns per-prompt pass rates and metrics.

    Scores the extracted <answer> span when present, otherwise the raw
    completion (many reasoning-gym scorers handle full text).
    """
    pass_rates = []
    all_scores = []
    for entry, completions in zip(entries, completions_per_prompt):
        n_pass = 0
        for completion in completions:
            answer = extract_answer(completion)
            candidate = answer if answer is not None else completion
            try:
                score = float(dataset.score_answer(answer=candidate, entry=entry))
            except Exception:
                score = 0.0
            all_scores.append(score)
            if score >= pass_threshold - 1e-9:
                n_pass += 1
        pass_rates.append(n_pass / len(completions))

    n = len(pass_rates)
    return {
        "n_prompts": n,
        "samples_per_prompt": len(completions_per_prompt[0]) if n else 0,
        "mean_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "pass_at_1": sum(pass_rates) / n if n else 0.0,
        "pass_at_k": sum(1 for r in pass_rates if r > 0) / n if n else 0.0,
        "trainable_frac": sum(1 for r in pass_rates if 0 < r < 1) / n if n else 0.0,
        "pass_rates": pass_rates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--tasks", default=None, help="comma-separated task names")
    parser.add_argument("--all", action="store_true", help="sweep every registered task")
    parser.add_argument("--list", action="store_true", help="list registered tasks and exit")
    parser.add_argument("--size", type=int, default=50, help="prompts per task")
    parser.add_argument("--samples", type=int, default=32, help="completions per prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--pass-threshold", type=float, default=1.0)
    parser.add_argument("--out", default="rg_sweep_results")
    parser.add_argument("--save-samples", type=int, default=2,
                        help="example completions to save per task")
    args = parser.parse_args()

    import reasoning_gym
    from reasoning_gym.factory import DATASETS

    registered = sorted(DATASETS.keys())
    if args.list:
        print("\n".join(registered))
        return

    if args.all:
        tasks = registered
    elif args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in tasks if t not in DATASETS]
        if unknown:
            parser.error(f"unknown tasks: {unknown} (see --list)")
    else:
        tasks = [t for t in DEFAULT_TASKS if t in DATASETS]
        skipped = [t for t in DEFAULT_TASKS if t not in DATASETS]
        if skipped:
            print(f"warning: skipping unregistered default tasks: {skipped}")

    generate = generate_vllm(args) if args.backend == "vllm" else generate_hf(args)

    os.makedirs(args.out, exist_ok=True)
    results = {}
    examples = {}
    start = time.time()
    for i, task in enumerate(tasks):
        try:
            dataset = reasoning_gym.create_dataset(task, size=args.size, seed=args.seed)
            entries = [dataset[j] for j in range(len(dataset))]
            conversations = build_conversations(entries)
            completions = generate(conversations)
            results[task] = score_task(
                dataset, entries, completions, args.pass_threshold
            )
            examples[task] = [
                {"question": entries[j]["question"], "completion": completions[j][0]}
                for j in range(min(args.save_samples, len(entries)))
            ]
            r = results[task]
            print(
                f"[{i + 1}/{len(tasks)}] {task}: mean_score={r['mean_score']:.3f} "
                f"pass@1={r['pass_at_1']:.3f} pass@{args.samples}={r['pass_at_k']:.3f} "
                f"trainable={r['trainable_frac']:.3f}"
            )
        except Exception as e:  # noqa: BLE001 - keep sweeping past broken tasks
            results[task] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{i + 1}/{len(tasks)}] {task}: ERROR {e}")

    summary = {
        "config": {k: v for k, v in vars(args).items()},
        "elapsed_sec": round(time.time() - start, 1),
        "results": results,
    }
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "examples.json"), "w") as f:
        json.dump(examples, f, indent=2)

    scored = [(t, r) for t, r in results.items() if "error" not in r]
    scored.sort(key=lambda x: x[1]["pass_at_1"], reverse=True)
    print(f"\n{'task':<40} {'mean':>6} {'p@1':>6} {'p@k':>6} {'train':>6}")
    for task, r in scored:
        print(
            f"{task:<40} {r['mean_score']:>6.3f} {r['pass_at_1']:>6.3f} "
            f"{r['pass_at_k']:>6.3f} {r['trainable_frac']:>6.3f}"
        )
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
