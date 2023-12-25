#!/bin/bash
set -x 
set -e

export offset=$1

# python evaluate_overgenerated_data.py prompt_way=own batch_size=48 offset=$offset data_dir=results_n_steps_500_vicuna data_prefix="individual_behaviors_vicuna_gcg_offset\{offset\}.json" target_lm=vicuna-chat adv_prompt_steps_per_instances=500
python evaluate_overgenerated_data.py prompt_way=own batch_size=48 offset=$offset data_dir=/fs/ess/PAA0201/lzy37ld/why_attack/data/results_n_steps_1000_llama2-chat data_prefix="individual_behaviors_llama2-chat_gcg_offset\{offset\}.json" target_lm=llama2-chat adv_prompt_steps_per_instances=1000
