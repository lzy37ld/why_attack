#!/bin/bash
set -x 
set -e


python check_tokens.py data="'/home/liao.629/why_attack/s_p_t_evaluate/promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl'"
python check_tokens.py data="'/home/liao.629/why_attack/s_p_t_evaluate/promptway_own|targetlm_do_sample_False|append_label_length_3.jsonl'"
python check_tokens.py data="'/home/liao.629/why_attack/s_p_t_evaluate/promptway_own|targetlm_do_sample_False|append_label_length_6.jsonl'"
python check_tokens.py data="'/home/liao.629/why_attack/s_p_t_evaluate/promptway_own|targetlm_do_sample_True|append_label_length_-1.jsonl'"
python check_tokens.py data="'/home/liao.629/why_attack/s_p_t_evaluate/promptway_own|targetlm_do_sample_True|append_label_length_3.jsonl'"
python check_tokens.py data="'/home/liao.629/why_attack/s_p_t_evaluate/promptway_own|targetlm_do_sample_True|append_label_length_6.jsonl'"