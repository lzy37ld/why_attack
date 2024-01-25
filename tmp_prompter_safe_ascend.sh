#!/bin/bash
set -x 
set -e

export WANDB_ENTITY=lzy37ld
export WANDB_PROJECT=attack_prompter


split_path=""
sampled_queries=""

# adv_train_model=meta-llama/Llama-2-7b-chat-hf
adv_train_model=lmsys/vicuna-7b-v1.5
# victim_model="llama2-7b-chat"
victim_model="vicuna-7b-chat-v1.5"
# prompt_type="llama2-chat_q_p"
prompt_type="vicuna-chat_q_p"
# make sure keep these two consistent

# step | loss_100 | random
sample_way_and_n_sample="loss_100_nsample=200"
split_path="data/train_val_test.json"
# step | loss_100 | random
sampled_queries="data/success_JB_victimmodel=${victim_model}_sampleway=${sample_way_and_n_sample}.json"
num_train_epochs=1

# num_queries=-1 by default
num_queries=-1

# default save_steps is 5000
save_steps=100000000

# default use_split_in_data=false
use_split_in_data=false

# default ppl_ratio=0.1
ppl_ratio=0.1
ppl_loss=false

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


# since victim model and unlearned model is consistent, so i use the same variable here.
output_dir=adv_train_model=${victim_model}_victim=${victim_model}_prompt_type=${prompt_type}_sample_way_and_n_sample=${sample_way_and_n_sample}_epoch_${num_train_epochs}
if [ "$ppl_loss" = true ]; then
    output_dir="${output_dir}_ppl_ratio=${ppl_ratio}"
fi
if [ $num_queries -gt 0 ]; then
    output_dir="${output_dir}_num_queries=${num_queries}"
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


torchrun --nproc_per_node=2 --master_port=1228 train_prompter_safe.py \
    --model_name_or_path ${adv_train_model} \
    --sampled_queries ${sampled_queries} \
    --split_path ${split_path} \
    --prompt_type ${prompt_type} \
    --use_split_in_data ${use_split_in_data} \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_steps $save_steps \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --tf32 True \
	  --report_to wandb \
    --save_only_model True \
    --num_queries $num_queries


    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \