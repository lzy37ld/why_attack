#!/bin/bash
set -x 
set -e

export WANDB_ENTITY=lzy37ld
export WANDB_PROJECT=attack_prompter
prompt_type="q_r_p"
train_ratio=0.8
model_name=llama2-base
export WANDB_NAME=vicuna_${prompt_type}_train_ratio_${train_ratio}_model_name_${model_name}

base_ckpt=$WHY_ATTACK_CKPT
base_data=$WHY_ATTACK_DATA

if [ -z "$base_ckpt" ]; then
    base_ckpt='.'
fi

if [ -z "$base_data" ]; then
    base_data='.'
fi

output_dir=$base_ckpt/prompter_vicuna_ckpt_${model_name}_${prompt_type}/


# ********************************************************************************************************************************************
# torchrun --nproc_per_node=8 --master_port=1234 train_prompter.py \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --data_path data/vicuna_process_100.json \
#     --bf16 True \
#     --output_dir $output_dir \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 2 \
#     --load_best_model_at_end True \
#     --save_total_limit 2 \
#     --save_only_model True \
#     --evaluation_strategy 'steps' \
#     --eval_steps 3000 \
#     --save_strategy 'steps' \
#     --save_steps 2500 \
#     --learning_rate 5e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --tf32 True \
# 	--report_to wandb \
# 	--prompt_type $prompt_type \
# 	--train_ratio $train_ratio

# ********************************************************************************************************************************************
# no evaluation
echo "no evaluation"

torchrun --nproc_per_node=8 --master_port=1234 train_prompter.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path data/vicuna_process_100.json \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_total_limit 2 \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_only_model True \
    --save_steps 2500 \
    --learning_rate 5e-5 \
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

