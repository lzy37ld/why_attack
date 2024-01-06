#!/bin/bash
set -x 
set -e


# ************************************************************************************************************************************************************************************************************************************************
# validation normal prompter
# ************************************************************************************************************************************************************************************************************************************************

# llama2-chat-7b
split="val"
checkpoints=(5000 10000 15000 20000 25000 30000 35000 final)
victim_model="llama2-7b-chat"
sample_way_and_n_sample_s=('random_nsample=200_epoch_5' 'step_nsample=200_epoch_5' 'loss_100_nsample=200_epoch_5')
for checkpoint in "${checkpoints[@]}"
do
	for sample_way_and_n_sample in "${sample_way_and_n_sample_s[@]}"
	do
		model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}'"
		show_name="${split}_checkpoint"
		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}-${split}'"
		if [ "$checkpoint" == "final" ]; then
			model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}'"
			s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-40000-${split}'"
		fi
		python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_prefix.choice=long batch_size=2
		python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_prefix.choice=medium batch_size=4
		python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_prefix.choice=short batch_size=8
		python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_rep=6
		python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_rep=5
		python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_rep=4
		python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_rep=3
		python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
		python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p
		python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50
		python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k
		python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k prompter_lm.generation_configs.num_return_sequences=50
	done
done




# ************************************************************************************************************************************************************************************************************************************************
# validation normal prompter
# ************************************************************************************************************************************************************************************************************************************************



# ************************************************************************************************************************************************************************************************************************************************
# vicuna-7b-chat-v1.5
split="val"
checkpoints=(5000 10000 15000 20000 25000 30000 35000 final)
victim_model="vicuna-7b-chat-v1.5"
sample_way_and_n_sample_s=('random_nsample=200_epoch_5' 'step_nsample=200_epoch_5' 'loss_100_nsample=200_epoch_5')
for checkpoint in "${checkpoints[@]}"
do
	for sample_way_and_n_sample in "${sample_way_and_n_sample_s[@]}"
	do
		model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}'"
		show_name="${split}_checkpoint"
		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-${checkpoint}-${split}'"
		if [ "$checkpoint" == "final" ]; then
			model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}'"
			s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=${sample_way_and_n_sample}/checkpoint-40000-${split}'"
		fi
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_prefix.choice=long batch_size=2
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_prefix.choice=medium batch_size=4
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_prefix.choice=short batch_size=8
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_rep=6
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_rep=5
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_rep=4
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_rep=3
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k prompter_lm.generation_configs.num_return_sequences=50
	done
done


# ************************************************************************************************************************************************************************************************************************************************
# gpt series


# split="val"
# checkpoints=(5000 10000 15000 20000 25000 30000 35000 final)
# victim_model="vicuna-7b-chat-v1.5"
# for checkpoint in "${checkpoints[@]}"
# do
#     model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-${checkpoint}'"
# 	show_name="${split}_checkpoint"
# 	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}'"
# 	if [ "$checkpoint" == "final" ]; then
#     	model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5'"
# 		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-40000-${split}'"
# 	fi
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_k
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_k prompter_lm.generation_configs.num_return_sequences=50
# done

# for checkpoint in "${checkpoints[@]}"
# do
#     model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-${checkpoint}'"
# 	show_name="${split}_checkpoint"
# 	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}'"
# 	if [ "$checkpoint" == "final" ]; then
#     	model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5'"
# 		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-40000-${split}'"
# 	fi
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_k
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_k prompter_lm.generation_configs.num_return_sequences=50
# done

# for checkpoint in "${checkpoints[@]}"
# do
#     model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=step_nsample=200_epoch_5/checkpoint-${checkpoint}'"
# 	show_name="${split}_checkpoint"
# 	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=step_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}'"
# 	if [ "$checkpoint" == "final" ]; then
#     	model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=step_nsample=200_epoch_5'"
# 		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=step_nsample=200_epoch_5/checkpoint-40000-${split}'"
# 	fi
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_k
# 	python evaluate_for_test_prompter.py target_lm=gpt3.5 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_k prompter_lm.generation_configs.num_return_sequences=50
# done








# for checkpoint in "${checkpoints[@]}"
# do
#     model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-${checkpoint}'"
# 	show_name="${split}_checkpoint"
# 	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}'"
# 	if [ "$checkpoint" == "final" ]; then
#     	model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5'"
# 		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-40000-${split}'"
# 	fi
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_k
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_k prompter_lm.generation_configs.num_return_sequences=50
# done

# for checkpoint in "${checkpoints[@]}"
# do
#     model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-${checkpoint}'"
# 	show_name="${split}_checkpoint"
# 	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}'"
# 	if [ "$checkpoint" == "final" ]; then
#     	model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5'"
# 		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-40000-${split}'"
# 	fi
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_k
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_k prompter_lm.generation_configs.num_return_sequences=50
# done

# for checkpoint in "${checkpoints[@]}"
# do
#     model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=step_nsample=200_epoch_5/checkpoint-${checkpoint}'"
# 	show_name="${split}_checkpoint"
# 	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=step_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}'"
# 	if [ "$checkpoint" == "final" ]; then
#     	model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=step_nsample=200_epoch_5'"
# 		s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=${victim_model}_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=step_nsample=200_epoch_5/checkpoint-40000-${split}'"
# 	fi
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_k
# 	python evaluate_for_test_prompter.py target_lm=gpt4 "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=true s_p_t_dir=$s_p_t_dir generation_configs@prompter_lm=top_k prompter_lm.generation_configs.num_return_sequences=50
# done