
#!/bin/bash

# 设置要搜索的目录
directory="/home/liao.629/why_attack/prompter_test_results"

# 使用 find 命令查找所有文件
# 然后通过循环读取每个文件的路径
find "$directory" -type f | while read file_path; do
    # 将文件路径赋值给变量
    current_file="$file_path"
	modified_string="${current_file/"why_attack"/"why_attack_lzy"}"
	# echo "$modified_string"
	path_before_last_slash="${modified_string%/*}"
	echo "$path_before_last_slash"
done
