#!/bin/bash

for file in '/users/PAA0201/lzy37ld/why_attack/s_p_t_evaluate/multi_1|max_new_tokens_60'/*
do
    python analysis.py --determine_way all --path $file --save_dir '/users/PAA0201/lzy37ld/why_attack/analysis/multi_1|max_new_tokens_100'
done

