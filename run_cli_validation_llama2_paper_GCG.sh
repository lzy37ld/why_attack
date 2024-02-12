#!/bin/bash
set -x 
set -e

# 
# First use val to select the best ckpt.
# Then use the selected one to try different decoding methods and tricks of reducing ppl. Log the ppl to do the ppl-defense exps.
# Third, they often show the training results. So I show the train results of that ckpt.
# For reducing ppl, I only choose q_rep, because when samplng 50 times, it's better than q_prefix
# 


# ************************************************************************************************************************************************************************************************************************************************
# validation normal prompter for one ckpt
# ************************************************************************************************************************************************************************************************************************************************

# # llama2-chat-7b
# split="test"
# ppl=true
# checkpoints=(30000)
# victim_model="llama2-7b-chat"
# sample_way_and_n_sample_s=('loss_100_nsample=200_epoch_5')
# for checkpoint in "${checkpoints[@]}"
# do
# 	for sample_way_and_n_sample in "${sample_way_and_n_sample_s[@]}"
# 	do
# 		model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}'"
# 		show_name="${split}_checkpoint"
# 		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}-${split}'"
# 		if [ "$checkpoint" == "final" ]; then
# 			model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}'"
# 			s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-40000-${split}'"
# 		fi
# 		# need to change the setting in config
# 		python evaluate_for_test_prompter.py target_lm=llama2-chat force_append=true prompt_way=own data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 prompt_own_list=config_GCG_suffix prompt_own_list_name=llama2_low
# 		python evaluate_for_test_prompter.py target_lm=llama2-chat force_append=true prompt_way=own data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 prompt_own_list=config_GCG_suffix prompt_own_list_name=llama2_all
# 		python evaluate_for_test_prompter.py target_lm=llama2-chat force_append=true prompt_way=own data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 sys_msg.choice=no_persuasive prompt_own_list=config_GCG_suffix prompt_own_list_name=llama2_low
# 		python evaluate_for_test_prompter.py target_lm=llama2-chat force_append=true prompt_way=own data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 sys_msg.choice=no_persuasive prompt_own_list=config_GCG_suffix prompt_own_list_name=llama2_all
# 	done
# done



# # transfer
# split="test"
# ppl=false
# checkpoints=(30000)
# victim_model="llama2-7b-chat"
# sample_way_and_n_sample_s=('loss_100_nsample=200_epoch_5')
# for checkpoint in "${checkpoints[@]}"
# do
# 	for sample_way_and_n_sample in "${sample_way_and_n_sample_s[@]}"
# 	do
# 		model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}'"
# 		show_name="${split}_checkpoint"
# 		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}-${split}-transfer'"
# 		if [ "$checkpoint" == "final" ]; then
# 			model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}'"
# 			s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-40000-${split}-transfer'"
# 		fi
# 		# need to change the setting in config
# 		python evaluate_for_test_prompter.py target_lm=vicuna-chat force_append=true prompt_way=own data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 prompt_own_list=config_GCG_suffix prompt_own_list_name=llama2_low
# 		python evaluate_for_test_prompter.py target_lm=vicuna-chat force_append=true prompt_way=own data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 prompt_own_list=config_GCG_suffix prompt_own_list_name=llama2_all
# 		python evaluate_for_test_prompter.py target_lm=mistral-instruct force_append=true prompt_way=own data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 prompt_own_list=config_GCG_suffix prompt_own_list_name=llama2_low
# 		python evaluate_for_test_prompter.py target_lm=mistral-instruct force_append=true prompt_way=own data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 prompt_own_list=config_GCG_suffix prompt_own_list_name=llama2_all
# 	done
# done




# # transfer
# transfer llama2 and vicuna
split="test"
ppl=false
checkpoints=(final)
victim_model="llama2-7b-chat_and_vicuna-7b-chat-v1.5"
sample_way_and_n_sample_s=('loss_100_nsample=200_epoch_3')
for checkpoint in "${checkpoints[@]}"
do
	for sample_way_and_n_sample in "${sample_way_and_n_sample_s[@]}"
	do
		model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}'"
		show_name="${split}_checkpoint"
		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}-${split}-transfer'"
		if [ "$checkpoint" == "final" ]; then
			model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}'"
			s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-final-${split}-transfer'"
		fi
		# need to change the setting in config
		python evaluate_for_test_prompter.py target_lm=llama2-chat force_append=true prompt_way=own data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 prompt_own_list=config_GCG_suffix prompt_own_list_name=llama2_vicuna_all
		python evaluate_for_test_prompter.py target_lm=vicuna-chat force_append=true prompt_way=own data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 prompt_own_list=config_GCG_suffix prompt_own_list_name=llama2_vicuna_all
		python evaluate_for_test_prompter.py target_lm=mistral-instruct force_append=true prompt_way=own data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 prompt_own_list=config_GCG_suffix prompt_own_list_name=llama2_vicuna_all
	done
done