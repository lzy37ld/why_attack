#!/bin/bash
set -x 
set -e

export WANDB_ENTITY=lzy37ld
export WANDB_PROJECT=attack_prompter


output_dir='prompter_ckpt'
prompt_type="q_r_p"
train_ratio=0.6
export WANDB_NAME=${prompt_type}_train_ratio_${train_ratio}


torchrun --nproc_per_node=4 --master_port=1234 train_prompter.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf \
    --data_path data/vicuna_process_100.json \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
	--eval_steps 50
    --save_strategy "steps" \
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
	--train_ratio $train_ratio \
    --data_path 'data/vicuna_process_100.json'
