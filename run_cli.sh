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

# python evaluate_overgenerated_data.py prompt_way=own batch_size=48 offset=$offset data_dir=results_n_steps_500_vicuna data_prefix="individual_behaviors_vicuna_gcg_offset\{offset\}.json" target_lm=vicuna-chat adv_prompt_steps_per_instances=500
python evaluate_overgenerated_data.py prompt_way=own batch_size=48 offset=$offset data_dir=/fs/ess/PAA0201/lzy37ld/why_attack/data/results_n_steps_1000_llama2-chat data_prefix="individual_behaviors_llama2-chat_gcg_offset\{offset\}.json" target_lm=llama2-chat adv_prompt_steps_per_instances=1000



# filter_overgenerated_data

python filter_overgenerated_data.py evaluated_data_path_template="/users/PAA0201/lzy37ld/why_attack/s_p_t_evaluate/llama2-7b-chat|max_new_tokens_60/\{offset\}|promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl" evaluated_model=llama2-7b-chat n_sample=200 sample_way=step
python filter_overgenerated_data.py evaluated_data_path_template="/users/PAA0201/lzy37ld/why_attack/s_p_t_evaluate/llama2-7b-chat|max_new_tokens_60/\{offset\}|promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl" evaluated_model=llama2-7b-chat n_sample=200 sample_way=random







# evaluate for prompter
python evaluate_for_test_prompter.py target_lm=vicuna-chat target_lm.generation_configs.max_new_tokens=100 data_args.prompt_type=q_target_p
python evaluate_for_test_prompter.py target_lm=vicuna-chat target_lm.generation_configs.max_new_tokens=100 data_args.prompt_type=q_target_lm_generation_p

python evaluate_for_test_prompter.py target_lm=llama2-chat target_lm.generation_configs.max_new_tokens=100 data_args.prompt_type=q_target_p
python evaluate_for_test_prompter.py target_lm=llama2-chat target_lm.generation_configs.max_new_tokens=100 data_args.prompt_type=q_target_lm_generation_p

python evaluate_for_test_prompter.py target_lm=mistral-instruct target_lm.generation_configs.max_new_tokens=100 data_args.prompt_type=q_target_p
python evaluate_for_test_prompter.py target_lm=mistral-instruct target_lm.generation_configs.max_new_tokens=100 data_args.prompt_type=q_target_lm_generation_p

python evaluate_for_test_prompter.py target_lm=llama2 target_lm.generation_configs.max_new_tokens=100 data_args.prompt_type=q_target_p
python evaluate_for_test_prompter.py target_lm=llama2 target_lm.generation_configs.max_new_tokens=100 data_args.prompt_type=q_target_lm_generation_p



# train prompter

export WANDB_ENTITY=lzy37ld
export WANDB_PROJECT=attack_prompter


output_dir='prompter_ckpt'
prompt_type="q_r_p"
train_ratio=0.6
export WANDB_NAME=${prompt_type}_train_ratio_${train_ratio}


torchrun --nproc_per_node=4 --master_port=1234 train_prompter.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path data/vicuna_process_100.json \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy 'steps' \
	--eval_steps 50 \
    --save_strategy 'steps' \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
	--report_to wandb \
	--prompt_type $prompt_type \
	--train_ratio $train_ratio