#!/bin/bash

for file in '/users/PAA0201/lzy37ld/why_attack/s_p_t_evaluate/multi_1|max_new_tokens_60'/*
do
    python analysis.py --determine_way all --path $file --save_dir '/users/PAA0201/lzy37ld/why_attack/analysis/multi_1|max_new_tokens_100'
done


python analysis.py --determine_way all --path '/home/liao.629/why_attack/s_p_t_evaluate/test|Yi-6B-Chat|max_new_tokens_60/promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl' --save_dir '/home/liao.629/why_attack/analysis/test|Yi|max_new_tokens_60'