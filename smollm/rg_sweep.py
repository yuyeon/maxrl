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
import time

from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEFAULT_SYSTEM_PROMPT_KEY = "default"

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

# reasoning-gym ships its own recommended system prompts (all establishing the
# <answer></answer> convention) and an extract_answer helper. We use those
# directly so the sweep prompts exactly like the library — and like the
# published reasoning-gym baselines — instead of a bespoke prompt, and so the
# same prompt can be carried verbatim into RL preprocessing. See
# reasoning_gym.utils.SYSTEM_PROMPTS ('default', 'simple', 'direct',
# 'chain_of_draft', 'DeepSeekZero'); extract_answer pulls the last <answer> span.


def build_conversations(entries, system_prompt):
    return [
        [
            {"role": "system", "content": system_prompt},
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

    tokenizer = llm.get_tokenizer()

    def render(conv):
        return tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )

    return generate, render


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

    def render(conv):
        return tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )

    return generate, render


def score_task(dataset, entries, completions_per_prompt, pass_threshold):
    """Score all completions; return (metrics, examples).

    Scores the extracted <answer> span when present, otherwise the raw
    completion (many reasoning-gym scorers handle full text). Alongside the
    aggregate metrics it captures the first passing and first failing
    completion seen, which are far more diagnostic than arbitrary samples.
    """
    pass_rates = []
    all_scores = []
    first_success = None
    first_failure = None
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
            record = {
                "question": entry["question"],
                "extracted_answer": answer,
                "score": score,
                "completion": completion,
            }
            if score >= pass_threshold - 1e-9:
                n_pass += 1
                if first_success is None:
                    first_success = record
            elif first_failure is None:
                first_failure = record
        pass_rates.append(n_pass / len(completions))

    n = len(pass_rates)
    metrics = {
        "n_prompts": n,
        "samples_per_prompt": len(completions_per_prompt[0]) if n else 0,
        "mean_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "pass_at_1": sum(pass_rates) / n if n else 0.0,
        "pass_at_k": sum(1 for r in pass_rates if r > 0) / n if n else 0.0,
        "trainable_frac": sum(1 for r in pass_rates if 0 < r < 1) / n if n else 0.0,
        "pass_rates": pass_rates,
    }
    examples = {"first_success": first_success, "first_failure": first_failure}
    return metrics, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument(
        "--system-prompt", default=DEFAULT_SYSTEM_PROMPT_KEY,
        choices=sorted(SYSTEM_PROMPTS.keys()),
        help="reasoning-gym system prompt to use (from reasoning_gym.utils.SYSTEM_PROMPTS)",
    )
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
    args = parser.parse_args()

    import reasoning_gym
    from reasoning_gym.factory import DATASETS

    registered = sorted(DATASETS.keys())
    if args.list:
        print("\n".join(registered))
        return

    if args.all:
        # 'composite' is a meta-dataset (combines other tasks) and cannot be
        # instantiated with defaults
        tasks = [t for t in registered if t != "composite"]
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

    system_prompt = SYSTEM_PROMPTS[args.system_prompt]
    print(f"system prompt: '{args.system_prompt}' from reasoning_gym.utils.SYSTEM_PROMPTS")
    print(f"---\n{system_prompt}\n---")

    generate, render = generate_vllm(args) if args.backend == "vllm" else generate_hf(args)

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.json")
    examples_path = os.path.join(args.out, "examples.json")
    progress_path = os.path.join(args.out, "progress.jsonl")

    def checkpoint(results, examples, elapsed):
        """Atomically rewrite summary/examples so a mid-sweep crash keeps data."""
        summary = {
            "config": {k: v for k, v in vars(args).items()},
            "elapsed_sec": round(elapsed, 1),
            "completed": len(results),
            "total": len(tasks),
            "results": results,
        }
        for path, obj in ((summary_path, summary), (examples_path, examples)):
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=2)
            os.replace(tmp, path)

    results = {}
    examples = {}
    start = time.time()
    # Fresh progress log for this run (append one JSON line per finished task).
    with open(progress_path, "w") as f:
        f.write(json.dumps({"event": "start", "total": len(tasks),
                            "config": {k: v for k, v in vars(args).items()}}) + "\n")
    for i, task in enumerate(tasks):
        try:
            dataset = reasoning_gym.create_dataset(task, size=args.size, seed=args.seed)
            entries = [dataset[j] for j in range(len(dataset))]
            conversations = build_conversations(entries, system_prompt)
            full_prompt = render(conversations[0])
            print(f"\n{'=' * 70}\n[{i + 1}/{len(tasks)}] {task} — example full prompt "
                  f"(as the model sees it):\n{'-' * 70}\n{full_prompt}\n{'=' * 70}")
            completions = generate(conversations)
            metrics, task_examples = score_task(
                dataset, entries, completions, args.pass_threshold
            )
            results[task] = metrics
            examples[task] = {"full_prompt": full_prompt, **task_examples}
            r = results[task]
            print(
                f"[{i + 1}/{len(tasks)}] {task}: mean_score={r['mean_score']:.3f} "
                f"pass@1={r['pass_at_1']:.3f} pass@{args.samples}={r['pass_at_k']:.3f} "
                f"trainable={r['trainable_frac']:.3f}"
            )
        except Exception as e:  # noqa: BLE001 - keep sweeping past broken tasks
            results[task] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{i + 1}/{len(tasks)}] {task}: ERROR {e}")

        # Persist after every task so an interrupted sweep loses at most one task.
        checkpoint(results, examples, time.time() - start)
        with open(progress_path, "a") as f:
            row = {"i": i + 1, "task": task, **results[task]}
            row.pop("pass_rates", None)  # keep the streaming log compact
            f.write(json.dumps(row) + "\n")

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
