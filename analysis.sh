#!/bin/bash

# 替换 '/path/to/directory' 为你想要遍历的文件夹路径
for file in /home/liao.629/rl-prompt-lzy/examples/text-style-transfer/s_p_t_evaluate/*
do
    python analysis.py --path $file
done



