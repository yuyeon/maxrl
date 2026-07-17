#!/bin/bash
# Inference sweep of SmolLM2-135M-Instruct across ALL reasoning-gym tasks.
# Intended for a single GPU machine (e.g. GH200); takes roughly 30-90 min.
#
# Reports per task: mean score, pass@1, pass@k, and the fraction of prompts
# in the trainable band (0 < pass rate < 1) — the tasks where RL training
# (maxrl/grpo/rloo) can get gradient signal.

set -e

MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
OUT=rg_sweep_results/smollm2-135m-instruct
SIZE=50        # prompts per task
SAMPLES=32     # completions per prompt (pass@32 resolution)
SEED=42

python3 -c "import reasoning_gym" 2>/dev/null || pip install reasoning-gym
python3 -c "import vllm" 2>/dev/null || {
  echo "vllm not found. On GH200 (aarch64) there are no PyPI wheels:"
  echo "build vllm from source or run inside an arm64 vLLM/NGC container."
  exit 1
}

python3 "$(dirname "$0")/rg_sweep.py" \
  --backend vllm \
  --model ${MODEL} \
  --all \
  --size ${SIZE} \
  --samples ${SAMPLES} \
  --seed ${SEED} \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-new-tokens 1024 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --out ${OUT}

echo
echo "Results: ${OUT}/summary.json (metrics) and ${OUT}/examples.json (sample completions)"
echo "Shortlist tasks with trainable_frac > 0 as RL candidates."
