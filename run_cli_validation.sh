#!/bin/bash
set -x 
set -e


# ************************************************************************************************************************************************************************************************************************************************
# validation
# ************************************************************************************************************************************************************************************************************************************************


split="val"
checkpoints=(5000 10000 15000 20000 25000 30000 35000)
for checkpoint in "${checkpoints[@]}"
do
    model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-${checkpoint}'"
	show_name="${split}_checkpoint"
	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}'"

	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k prompter_lm.generation_configs.num_return_sequences=50
done


checkpoints=(5000 10000 15000 20000 25000 30000 35000)
for checkpoint in "${checkpoints[@]}"
do
    model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-${checkpoint}'"
	show_name="${split}_checkpoint"
	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}'"

	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k prompter_lm.generation_configs.num_return_sequences=50
done


checkpoints=(5000 10000 15000 20000 25000 30000 35000)
for checkpoint in "${checkpoints[@]}"
do
    model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=step_nsample=200_epoch_5/checkpoint-${checkpoint}'"
	show_name="${split}_checkpoint"
	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=step_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}'"

	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_p prompter_lm.generation_configs.num_return_sequences=50
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k
	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100 generation_configs@prompter_lm=top_k prompter_lm.generation_configs.num_return_sequences=50
done




# ************************************************************************************************************************************************************************************************************************************************
# train
# ************************************************************************************************************************************************************************************************************************************************


# split="train"
# checkpoints=(5000 10000 15000 20000 25000 30000 35000)

# for checkpoint in "${checkpoints[@]}"
# do
#     model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-${checkpoint}'"
# 	show_name="${split}_checkpoint"
# 	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}'"

# 	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
# done


# checkpoints=(5000 10000 15000 20000 25000 30000 35000)
# for checkpoint in "${checkpoints[@]}"
# do
#     model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-${checkpoint}'"
# 	show_name="${split}_checkpoint"
# 	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=loss_100_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}'"

# 	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
# done





# checkpoints=(5000 10000 15000 20000 25000 30000 35000)
# for checkpoint in "${checkpoints[@]}"
# do
#     model_name="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=step_nsample=200_epoch_5/checkpoint-${checkpoint}'"
# 	show_name="${split}_checkpoint"
# 	s_p_t_dir="'/fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=step_nsample=200_epoch_5/checkpoint-${checkpoint}-${split}'"

# 	python evaluate_for_test_prompter.py target_lm=llama2-chat "prompter_lm.model_name=$model_name" "prompter_lm.show_name=$show_name" data_args.split=${split} ppl=false s_p_t_dir=$s_p_t_dir target_lm.generation_configs.max_new_tokens=100
# done

