#!/bin/bash
set -x 
set -e

checkpoints=(5000 10000 15000 20000 25000)

for checkpoint in "${checkpoints[@]}"
do
    model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-${checkpoint}'"
	show_name="val_checkpoint"
	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-${checkpoint}-val'"

	# evaluate for prompter
	# only need for one victim model
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=val ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
done


checkpoints=(5000 10000 15000 20000 25000 30000)
for checkpoint in "${checkpoints[@]}"
do
    model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-${checkpoint}'"
	show_name="val_checkpoint"
	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-${checkpoint}-val'"

	# evaluate for prompter
	# only need for one victim model
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=val ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
done








# validation only need for one victim_model, which is himself
# python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=val ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
# python evaluate_for_test_prompter.py target_lm=mistral-instruct "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=val ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100


# validation part only need greedy setting.
# python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=val ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p