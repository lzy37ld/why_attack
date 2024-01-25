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



# ************************************************************************************************************************************************************************************************************************************************
# vicuna-7b-chat-v1.5
split="test"
ppl=true
checkpoints=(15000)
victim_model="vicuna-7b-chat-v1.5"
sample_way_and_n_sample_s=('loss_100_nsample=200_epoch_5')
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
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_prefix.choice=long
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_prefix.choice=medium
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_prefix.choice=short
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_rep=6
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_rep=5
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_rep=4
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50 q_rep=3
		
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=100 q_rep=6
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=100 q_rep=5
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=100 q_rep=4
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=100 q_rep=3

		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=400 q_rep=3
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=400 q_rep=4
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=200 q_rep=3 "q_s_position='prompter_lm=raw|target_lm=processed'"
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=200 q_rep=4 "q_s_position='prompter_lm=raw|target_lm=processed'"
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=400 q_rep=4 "q_s_position='prompter_lm=raw|target_lm=processed'"
		
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k
		python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=${ppl} s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k prompter_lm.generation_configs.num_return_sequences=50
	done
done

