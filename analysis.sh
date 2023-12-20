#!/bin/bash

# for file in '/users/PAA0201/lzy37ld/why_attack/s_p_t_evaluate/multi_1|max_new_tokens_60'/*
# do
#     python analysis.py --determine_way all --path $file --save_dir '/users/PAA0201/lzy37ld/why_attack/analysis/multi_1|max_new_tokens_100'
# done


name='prompter_prompter_vicuna_ckpt_llama2-base_q_r_p|promptway_prompter_q_target_lm_generation_p/llama2-7b-chat|max_new_tokens_100'
python analysis.py --determine_way all --path "/home/liao.629/why_attack/prompter_test_results/${name}/targetlm_do_sample_False|append_label_length_-1.jsonl" --save_dir "/home/liao.629/why_attack/analysis_prompter/${name}"

name='prompter_prompter_vicuna_ckpt_llama2-base_q_r_p|promptway_prompter_q_target_lm_generation_p/llama2-7b|max_new_tokens_100'
python analysis.py --determine_way all --path "/home/liao.629/why_attack/prompter_test_results/${name}/targetlm_do_sample_False|append_label_length_-1.jsonl" --save_dir "/home/liao.629/why_attack/analysis_prompter/${name}"



name='prompter_prompter_vicuna_ckpt_llama2-base_q_r_p|promptway_prompter_q_target_lm_generation_p/Mistral-7B-Instruct-v0.1|max_new_tokens_100'
python analysis.py --determine_way all --path "/home/liao.629/why_attack/prompter_test_results/${name}/targetlm_do_sample_False|append_label_length_-1.jsonl" --save_dir "/home/liao.629/why_attack/analysis_prompter/${name}"


name='prompter_prompter_vicuna_ckpt_llama2-base_q_r_p|promptway_prompter_q_target_lm_generation_p/vicuna-7b-chat-v1.5|max_new_tokens_100'
python analysis.py --determine_way all --path "/home/liao.629/why_attack/prompter_test_results/${name}/targetlm_do_sample_False|append_label_length_-1.jsonl" --save_dir "/home/liao.629/why_attack/analysis_prompter/${name}"



name='prompter_prompter_vicuna_ckpt_llama2-base_q_r_p|promptway_prompter_q_target_p/llama2-7b-chat|max_new_tokens_100'
python analysis.py --determine_way all --path "/home/liao.629/why_attack/prompter_test_results/${name}/targetlm_do_sample_False|append_label_length_-1.jsonl" --save_dir "/home/liao.629/why_attack/analysis_prompter/${name}"

name='prompter_prompter_vicuna_ckpt_llama2-base_q_r_p|promptway_prompter_q_target_p/llama2-7b|max_new_tokens_100'
python analysis.py --determine_way all --path "/home/liao.629/why_attack/prompter_test_results/${name}/targetlm_do_sample_False|append_label_length_-1.jsonl" --save_dir "/home/liao.629/why_attack/analysis_prompter/${name}"



name='prompter_prompter_vicuna_ckpt_llama2-base_q_r_p|promptway_prompter_q_target_p/Mistral-7B-Instruct-v0.1|max_new_tokens_100'
python analysis.py --determine_way all --path "/home/liao.629/why_attack/prompter_test_results/${name}/targetlm_do_sample_False|append_label_length_-1.jsonl" --save_dir "/home/liao.629/why_attack/analysis_prompter/${name}"


name='prompter_prompter_vicuna_ckpt_llama2-base_q_r_p|promptway_prompter_q_target_p/vicuna-7b-chat-v1.5|max_new_tokens_100'
python analysis.py --determine_way all --path "/home/liao.629/why_attack/prompter_test_results/${name}/targetlm_do_sample_False|append_label_length_-1.jsonl" --save_dir "/home/liao.629/why_attack/analysis_prompter/${name}"