#!/bin/bash

for file in /home/liao.629/why_attack/s_p_t_evaluate/max_new_tokens_100/*
do
    python analysis.py --determine_way all --path $file --save_dir /home/liao.629/why_attack/analysis/max_new_tokens_100
done



