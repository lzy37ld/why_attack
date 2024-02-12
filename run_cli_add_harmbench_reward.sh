#!/bin/bash

set -e
set -x

# 设定要遍历的顶级目录
# top_dir="/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-30000-test"
# top_dir="/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-30000-test-transfer"
top_dir="/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=vicuna-7b-chat-v1.5_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-15000-test"
# top_dir="/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=vicuna-7b-chat-v1.5_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-15000-test-transfer"

# 使用find命令查找所有.jsonl文件，并通过for循环遍历它们
find "$top_dir" -type f -name "*.jsonl" | while read file; do
  python add_reward_for_harmbench.py "path='$file'"
done
