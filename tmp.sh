#!/bin/bash
set -x 
set -e

export offset=$1

# python evaluate_for_each_instances.py prompt_way=own batch_size=48 target_lm=llama2 offset=$offset
python evaluate_for_each_instances.py prompt_way=own batch_size=48 offset=$offset data_dir=results_n_steps_500_vicuna data_prefix="individual_behaviors_vicuna_gcg_offset\{offset\}.json" target_lm=vicuna-chat