#!/bin/bash
set -x 
set -e


# multi_0
prompway = no
python evaluate.py prompt_way=no append_label_length=-1
python evaluate.py prompt_way=no append_label_length=3
python evaluate.py prompt_way=no append_label_length=6

python evaluate.py prompt_way=no append_label_length=-1 target_lm.generation_configs.do_sample=True
python evaluate.py prompt_way=no append_label_length=3 target_lm.generation_configs.do_sample=True
python evaluate.py prompt_way=no append_label_length=6 target_lm.generation_configs.do_sample=True


# promptway = own
python evaluate.py prompt_way=own append_label_length=-1
python evaluate.py prompt_way=own append_label_length=3
python evaluate.py prompt_way=own append_label_length=6

python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.do_sample=True
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.do_sample=True
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.do_sample=True





# prompway = no
python evaluate.py prompt_way=no append_label_length=-1 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=no append_label_length=3 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=no append_label_length=6 target_lm.generation_configs.max_new_tokens=100

python evaluate.py prompt_way=no append_label_length=-1 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=no append_label_length=3 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=no append_label_length=6 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100


# promptway = own
python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.max_new_tokens=100

python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100






# multi_1
# max_new_tokens = 60

# promptway = own
python evaluate.py prompt_way=own append_label_length=-1 data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=3 data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=6 data=./data/mutli_1_reformatted_attack.jsonl multi=1

python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.do_sample=True data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.do_sample=True data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.do_sample=True data=./data/mutli_1_reformatted_attack.jsonl multi=1




# max_new_tokens = 100

# promptway = own
python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.max_new_tokens=100 data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.max_new_tokens=100 data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.max_new_tokens=100 data=./data/mutli_1_reformatted_attack.jsonl multi=1

python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100 data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100 data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100 data=./data/mutli_1_reformatted_attack.jsonl multi=1






python check_tokens.py data="'/home/liao.629/why_attack/s_p_t_evaluate/promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl'"
python check_tokens.py data="'/home/liao.629/why_attack/s_p_t_evaluate/promptway_no|targetlm_do_sample_False|append_label_length_-1.jsonl'"












# evaluate_overgenerated_data

python evaluate_overgenerated_data.py prompt_way=own batch_size=48 offset=$offset data_dir=/fs/ess/PAA0201/lzy37ld/why_attack_lzy/data/results_n_steps_500_vicuna data_prefix="individual_behaviors_vicuna_gcg_offset\{offset\}.json" target_lm=vicuna-chat adv_prompt_steps_per_instances=500 s_p_t_dir=/fs/ess/PAA0201/lzy37ld/why_attack/data/s_p_t_evaluate
python evaluate_overgenerated_data.py prompt_way=own batch_size=48 offset=$offset data_dir=/fs/ess/PAA0201/lzy37ld/why_attack_lzy/data/results_n_steps_1000_llama2-chat data_prefix="individual_behaviors_llama2-chat_gcg_offset\{offset\}.json" target_lm=llama2-chat adv_prompt_steps_per_instances=1000



# filter_overgenerated_data

python filter_overgenerated_data.py evaluated_data_path_template="/fs/ess/PAA0201/lzy37ld/why_attack/data/s_p_t_evaluate/llama2-7b-chat|max_new_tokens_60/\{offset\}|promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl" evaluated_model=llama2-7b-chat n_sample=200 sample_way=step +interval=192000
python filter_overgenerated_data.py evaluated_data_path_template="/fs/ess/PAA0201/lzy37ld/why_attack/data/s_p_t_evaluate/llama2-7b-chat|max_new_tokens_60/\{offset\}|promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl" evaluated_model=llama2-7b-chat n_sample=200 sample_way=random +interval=192000
python filter_overgenerated_data.py evaluated_data_path_template="/fs/ess/PAA0201/lzy37ld/why_attack/data/s_p_t_evaluate/llama2-7b-chat|max_new_tokens_60/\{offset\}|promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl" evaluated_model=llama2-7b-chat n_sample=200 sample_way=loss_100 +interval=192000

python filter_overgenerated_data.py evaluated_data_path_template="/fs/ess/PAA0201/lzy37ld/why_attack/data/s_p_t_evaluate/vicuna-7b-chat-v1.5|max_new_tokens_60/\{offset\}|promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl" evaluated_model=vicuna-7b-chat-v1.5 n_sample=200 sample_way=step +interval=64000
python filter_overgenerated_data.py evaluated_data_path_template="/fs/ess/PAA0201/lzy37ld/why_attack/data/s_p_t_evaluate/vicuna-7b-chat-v1.5|max_new_tokens_60/\{offset\}|promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl" evaluated_model=vicuna-7b-chat-v1.5 n_sample=200 sample_way=random +interval=64000
python filter_overgenerated_data.py evaluated_data_path_template="/fs/ess/PAA0201/lzy37ld/why_attack/data/s_p_t_evaluate/vicuna-7b-chat-v1.5|max_new_tokens_60/\{offset\}|promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl" evaluated_model=vicuna-7b-chat-v1.5 n_sample=200 sample_way=loss_100 +interval=64000


python split_train_test_val.py "success_jb_path='data/success_JB_victimmodel=llama2-7b-chat_sampleway=loss_100_nsample=200.json'"


# train prompter
# train prompter
see_prompter_ascend.sh
# train prompter
# train prompter



# validation
# validation
see run_cli_validation.sh
# validation
# validation