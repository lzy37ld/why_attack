#!/bin/bash
set -x 
set -e


# only choose one...
best_ckpt="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-20000'"
model_name=$best_ckpt
show_name="'prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5_ckpt-20000'"
s_p_t_dir="./prompter_test_results"




python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=test ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=test ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p
python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=test ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k
python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=test ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=beam_search


python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=test ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=test ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p
python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=test ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k
python evaluate_for_test_prompter.py target_lm=vicuna-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=test ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=beam_search

python evaluate_for_test_prompter.py target_lm=mistral-instruct "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=test ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
python evaluate_for_test_prompter.py target_lm=mistral-instruct "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=test ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p
python evaluate_for_test_prompter.py target_lm=mistral-instruct "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=test ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k
python evaluate_for_test_prompter.py target_lm=mistral-instruct "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=test ppl=true s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=beam_search