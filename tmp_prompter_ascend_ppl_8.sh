#!/bin/bash
set -x 
set -e

export WANDB_ENTITY=lzy37ld
export WANDB_PROJECT=attack_prompter
prompt_type="q_p"
model_name=llama2-base

split_path=""
sampled_queries=""

victim_model="llama2-7b-chat"
sample_way_and_n_sample="loss_100_nsample=200"
split_path="data/train_val_test.json"
# step | loss_100 | random
sampled_queries="data/success_JB_victimmodel=${victim_model}_sampleway=${sample_way_and_n_sample}.json"
num_train_epochs=5
# default ppl_ratio=0.1
ppl_ratio=0.5
ppl_loss=true

if [[ $sampled_queries == *"$sample_way_and_n_sample"* ]]; then
  echo "'$sample_way_and_n_sample' is in '$sampled_queries'"
else
  echo "name dont follow schema"
  exit 1
fi

if [[ $sampled_queries == *"$victim_model"* ]]; then
  echo "'$victim_model' is in '$sampled_queries'"
else
  echo "name dont follow schema"
  exit 1
fi

output_dir=prompter_victim=${victim_model}_prompt_type=${prompt_type}_model_name=${model_name}_sample_way_and_n_sample=${sample_way_and_n_sample}_epoch_${num_train_epochs}
if [ "$ppl_loss" = true ]; then
    output_dir="${output_dir}_ppl_ratio=${ppl_ratio}"
fi
export WANDB_NAME=${output_dir}
base_ckpt=$WHY_ATTACK_CKPT
base_data=$WHY_ATTACK_DATA

if [ -z "$base_ckpt" ]; then
    base_ckpt='.'
fi

if [ -z "$base_data" ]; then
    base_data='.'
fi


output_dir=$base_ckpt/${output_dir}/

# no evaluation
echo "no evaluation"



# torchrun --nproc_per_node=4 --master_port=1234 train_prompter.py \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --sampled_queries ${sampled_queries} \
#     --split_path ${split_path} \
#     --bf16 True \
#     --output_dir $output_dir \
#     --num_train_epochs $num_train_epochs \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 2 \
#     --evaluation_strategy 'no' \
#     --save_strategy 'steps' \
#     --save_steps 5000 \
#     --learning_rate 5e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --tf32 True \
# 	  --report_to wandb \
# 	  --prompt_type $prompt_type \
#     --ppl_ratio $ppl_ratio \
#     --ppl_loss $ppl_loss



accelerate launch --config_file myconfig/ds_zero3_8.yaml --main_process_port 1231 train_prompter.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --sampled_queries ${sampled_queries} \
    --split_path ${split_path} \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_steps 5000 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
	  --report_to wandb \
	  --prompt_type $prompt_type \
    --ppl_ratio $ppl_ratio \
    --ppl_loss $ppl_loss