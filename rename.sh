#!/bin/bash

# 设置您的目标文件夹路径
DIRECTORY="/fs/ess/PAA0201/lzy37ld/why_attack/data/results_n_steps_500_two_vicuna"

# 遍历文件夹中的所有文件
for FILE in "$DIRECTORY"/*
do
    # 获取文件的新名称，通过替换指定的字符串
    NEW_NAME=$(echo "$FILE" | sed 's/two_vicuna_gcg_progressive/individual_behaviors_two_vicuna_gcg/g')

    # 重命名文件
    mv "$FILE" "$NEW_NAME"
done

echo "Renaming completed."
