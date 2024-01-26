#!/bin/bash
set -x 
set -e


# ************************************************************************************************************************************************************************************************************************************************
# gpt series
# ************************************************************************************************************************************************************************************************************************************************
# gpt series



# # llama2-chat-7b
# split="val"
# ppl=false
# checkpoints=(25000)
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

# 		python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 batch_size=8
# 		python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=100 batch_size=8
# 		python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=200 q_rep=4 batch_size=4
# 		python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=200 q_rep=4 "q_s_position='prompter_lm=raw|target_lm=processed'" batch_size=4
# 	done
# done



# # vicuna-7b-chat-v1.5
# split="val"
# ppl=false
# checkpoints=(15000)
# victim_model="vicuna-7b-chat-v1.5"
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

# 		python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 batch_size=8
# 		python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=100 batch_size=8
# 		python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=400 batch_size=8
# 		python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 batch_size=8 prompt_concat=3 num_prompt_group=100
# 		# python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=200 q_rep=4 batch_size=4
# 		# python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=200 q_rep=4 "q_s_position='prompter_lm=raw|target_lm=processed'" batch_size=4
# 	done
# done


# vicuna-7b-chat-v1.5_and_vicuna-13b-chat-v1.5_and_guanaco-7b-chat_and_guanaco-13b-chat"

# split="gpt_heldin_test"
# ppl=false
# checkpoints=(final)
# victim_model="vicuna-7b-chat-v1.5_and_vicuna-13b-chat-v1.5_and_guanaco-7b-chat_and_guanaco-13b-chat"
# sample_way_and_n_sample_s=('loss_100_nsample=200_epoch_3')
# for checkpoint in "${checkpoints[@]}"
# do
# 	for sample_way_and_n_sample in "${sample_way_and_n_sample_s[@]}"
# 	do
# 		model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}'"
# 		show_name="${split}_checkpoint"
# 		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}-${split}'"
# 		if [ "$checkpoint" == "final" ]; then
# 			model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}'"
# 			s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/final-${split}'"
# 		fi
# 		python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=400 batch_size=4 prompter_lm.batch_size=4 target_lm.batch_size=4 reward_lm.batch_size=4 w_affirm_suffix=true
# 		# python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=400 batch_size=4 prompter_lm.batch_size=4 target_lm.batch_size=4 reward_lm.batch_size=4 w_affirm_suffix=true
# 	done
# done


split="gpt_test"
ppl=false
checkpoints=(final)
victim_model="vicuna-7b-chat-v1.5_and_vicuna-13b-chat-v1.5_and_guanaco-7b-chat_and_guanaco-13b-chat"
sample_way_and_n_sample_s=('loss_100_nsample=200_epoch_3')
for checkpoint in "${checkpoints[@]}"
do
	for sample_way_and_n_sample in "${sample_way_and_n_sample_s[@]}"
	do
		model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}'"
		show_name="${split}_checkpoint"
		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}-${split}'"
		if [ "$checkpoint" == "final" ]; then
			model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}'"
			s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/final-${split}'"
		fi
		python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=250 batch_size=4 prompter_lm.batch_size=4 target_lm.batch_size=4 reward_lm.batch_size=4 w_affirm_suffix=true
		python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=400 batch_size=4 prompter_lm.batch_size=4 target_lm.batch_size=4 reward_lm.batch_size=4 w_affirm_suffix=true
		python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=250 batch_size=4 prompter_lm.batch_size=4 target_lm.batch_size=4 reward_lm.batch_size=4 w_affirm_suffix=true
		python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=400 batch_size=4 prompter_lm.batch_size=4 target_lm.batch_size=4 reward_lm.batch_size=4 w_affirm_suffix=true

		python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=250 batch_size=4 prompter_lm.batch_size=4 target_lm.batch_size=4 reward_lm.batch_size=4
		python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=400 batch_size=4 prompter_lm.batch_size=4 target_lm.batch_size=4 reward_lm.batch_size=4
		python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=250 batch_size=4 prompter_lm.batch_size=4 target_lm.batch_size=4 reward_lm.batch_size=4
		python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=400 batch_size=4 prompter_lm.batch_size=4 target_lm.batch_size=4 reward_lm.batch_size=4
	done
done