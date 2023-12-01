#!/bin/bash

for file in /home/liao.629/why_attack/s_p_t_evaluate/*
do
    python analysis.py --determine_way all --path $file
done



