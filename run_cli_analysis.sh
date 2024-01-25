#!/bin/bash
set -x 
set -e



split="val"
checkpoints=(5000 10000 15000 20000 25000 30000 35000 40000)
# checkpoints=(15000)
sample_ways=(loss_100 random step)
# sample_ways=(loss_100)
victim_model=llama2-7b-chat

for checkpoint in "${checkpoints[@]}"
do
	for sample_way in "${sample_ways[@]}"
	do
		directory="/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way}_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}"
		find "$directory" -type f | while read file_path; do
			# 将文件路径赋值给变量
			current_file="$file_path"
			modified_string="${current_file/"-${split}"/"-${split}_analysis"}"
			# echo "$modified_string"
			path_before_last_slash="${modified_string%/*}"
			# echo "$path_before_last_slash"
			python analysis.py --path ${current_file} --save_dir ${path_before_last_slash}
		done
	done
done




# split="train"
# checkpoints=(5000 10000 15000 20000 25000 30000 35000)
# sample_ways=(loss_100 random step)
# for checkpoint in "${checkpoints[@]}"
# do
# 	for sample_way in "${sample_ways[@]}"
#   do
# 		directory="/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way}_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}"
# 		find "$directory" -type f | while read file_path; do
# 			# 将文件路径赋值给变量
# 			current_file="$file_path"
# 			modified_string="${current_file/"-${split}"/"-${split}_analysis"}"
# 			# echo "$modified_string"
# 			path_before_last_slash="${modified_string%/*}"
# 			# echo "$path_before_last_slash"
# 			python analysis.py --path ${current_file} --save_dir ${path_before_last_slash}
# 		done
# 	done
# done







# directory="/users/PAA0201/lzy37ld/why_attack_lzy/prompter_test_results"
# find "$directory" -type f | while read file_path; do
# 	# 将文件路径赋值给变量
# 	current_file="$file_path"
# 	modified_string="${current_file/"prompter_test_results"/"prompter_test_results_analysis"}"
# 	# echo "$modified_string"
# 	path_before_last_slash="${modified_string%/*}"
# 	# echo "$path_before_last_slash"
# 	python analysis.py --path ${current_file} --save_dir ${path_before_last_slash}
# done


# directory="/users/PAA0201/lzy37ld/why_attack_lzy/prompter_hard_results"
# find "$directory" -type f | while read file_path; do
# 	# 将文件路径赋值给变量
# 	current_file="$file_path"
# 	modified_string="${current_file/"prompter_hard_results"/"prompter_hard_results_analysis"}"
# 	# echo "$modified_string"
# 	path_before_last_slash="${modified_string%/*}"
# 	# echo "$path_before_last_slash"
# 	python analysis.py --path ${current_file} --save_dir ${path_before_last_slash}
# done



# directory="/users/PAA0201/lzy37ld/why_attack_lzy/prompter_unknown_results"
# find "$directory" -type f | while read file_path; do
# 	# 将文件路径赋值给变量
# 	current_file="$file_path"
# 	modified_string="${current_file/"prompter_unknown_results"/"prompter_unknown_results_analysis"}"
# 	# echo "$modified_string"
# 	path_before_last_slash="${modified_string%/*}"
# 	# echo "$path_before_last_slash"
# 	python analysis.py --path ${current_file} --save_dir ${path_before_last_slash}
# done
