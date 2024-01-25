#!/bin/bash
set -x 
set -e

export offset=$1

python evaluate_overgenerated_data.py prompt_way=own batch_size=48 offset=$offset data_dir=/fs/ess/PAA0201/lzy37ld/why_attack/data/results_n_steps_500_four_vicuna_guanaco data_prefix="individual_behaviors_four_vicuna_guanaco_gcg_offset\{offset\}.json" target_lm=guanaco-chat adv_prompt_steps_per_instances=500 s_p_t_dir=/fs/ess/PAA0201/lzy37ld/why_attack/data/s_p_t_evaluate/four_vicuna_guanaco
