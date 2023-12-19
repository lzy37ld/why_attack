



#!/bin/bash

# 指定要检查的目录，你可以将这里的路径替换为你想要检查的目录
DIRECTORY="/fs/ess/PAA0201/lzy37ld/why_attack/data/results_n_steps_1000_llama2-chat"

# 指定行数不等于此值的文件将被显示
TARGET_LINE_COUNT=10280112

# 检查目录是否存在
if [ ! -d "$DIRECTORY" ]; then
    echo "目录不存在: $DIRECTORY"
    exit 1
fi

# 遍历目录下的每个文件
for FILE in "$DIRECTORY"/*
do
    if [ -f "$FILE" ]; then
        # 获取文件的行数
        LINE_COUNT=$(wc -l < "$FILE")

        # 如果行数不等于指定值，则输出
        if [ "$LINE_COUNT" -ne "$TARGET_LINE_COUNT" ]; then
            echo "$FILE: $LINE_COUNT"
        fi
    fi
done
