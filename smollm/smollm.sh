#!/bin/bash
# SmolLM2-360M Training Script

set -e

# ============ Configuration ============
MODEL_PATH=~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951/
TRAIN_DATA=$HOME/data/reasoning_gym/chain_sum/train.parquet
VAL_DATA=$HOME/data/reasoning_gym/chain_sum/val.parquet
CHECKPOINT_DIR=$HOME/checkpoints
PROJECT_NAME=MaxRL_SmolLM-360M-chain_sum

# Training hyperparameters
ADVANTAGE_ESTIMATOR=maxrl

# uncomment the following lines if you want to run GRPO or RLOO
# ADVANTAGE_ESTIMATOR=grpo
# ADVANTAGE_ESTIMATOR=rloo

TRUNCATE_ORDER=64
LR=1e-5
N_ROLLOUTS=128
N_VAL=32
MAX_RESPONSE_LENGTH=2048

EXPERIMENT_NAME=${ADVANTAGE_ESTIMATOR}_${N_ROLLOUTS}rollouts

# ============ Ray Setup ============
ray stop --force 2>/dev/null || true
ray start --head --num-gpus 1
ray status

# ============ Training ============
python3 -m verl.trainer.main_ppo \
  ray_init.ray_dir=/tmp/ray \
  algorithm.adv_estimator=${ADVANTAGE_ESTIMATOR} \
  algorithm.use_kl_in_reward=False \
  algorithm.pass_k=${TRUNCATE_ORDER} \
  algorithm.truncate_order=${TRUNCATE_ORDER} \
  data.train_files=${TRAIN_DATA} \
  data.val_files=${VAL_DATA} \
  data.train_batch_size=64 \
  data.filter_overlong_prompts=True \
  data.max_prompt_length=512 \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=${N_VAL} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  algorithm.kl_ctrl.kl_coef=0.0 \
  reward_model.reward_manager=multi_thread \
  +reward_model.reward_kwargs.num_reward_actors=64 \
  +reward_model.reward_kwargs.zero_reward_on_max_response_length=True \
  +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.logger=['console','wandb'] \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=100 \
  trainer.default_local_dir=${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
  trainer.total_epochs=200
