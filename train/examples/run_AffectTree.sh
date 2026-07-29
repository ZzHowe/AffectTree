#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
DATA_PATH=~/data/custom_dataset/videomme_dataset_output/videomme_dataset_complete.jsonl
DATA_PATH=~/data/custom_dataset/videomme_with_cache/raw_videomme_dataset_complete.jsonl


python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_PATH} \
    data.val_files=${DATA_PATH} \
    data.rollout_batch_size=8 \
    data.val_batch_size=8 \
    worker.actor.global_batch_size=8 \
    data.prompt_key=full_question \
    data.answer_key=ground_truth_answer \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=./examples/reward_function/video.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=8
