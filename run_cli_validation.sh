#!/bin/bash
set -x 
set -e


model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200/checkpoint-10000'"
show_name="val_checkpoint"
s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200/checkpoint-10000-val'"

# evaluate for prompter
python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=val ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=val ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
python evaluate_for_test_prompter.py target_lm=mistral-instruct "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=val ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=val ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p
python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=val ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p
python evaluate_for_test_prompter.py target_lm=mistral-instruct "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=val ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p