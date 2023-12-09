#!/bin/bash
set -x 
set -e

# export offset=$1

# python evaluate_for_each_instances.py prompt_way=own batch_size=48 target_lm=llama2 offset=$offset


for file in /home/liao.629/why_attack/results_n_steps_500/*; do
  mv "$file" "${file/llama2_gcg/llama2-7b-chat_gcg}"
done
