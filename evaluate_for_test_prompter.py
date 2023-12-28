import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedTokenizer
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from reward_target_lm import create_reward,create_targetlm,create_prompterlm
from torch.utils.data import DataLoader
import jsonlines
import os
from tqdm import tqdm
from pathlib import Path
from accelerate.utils import set_seed
from print_color import print
import json
import time
from torch.utils.data import Dataset
from itertools import chain, repeat



PROMPT_DICT = {
    # "q_p_r": (
    #     "### Query:{query}### Prompt:{prompt}### Response:{response}"
    # ),
    "q_target_lm_generation_p": (
       "### Query:{q} ### Response:{target_lm_generation} ### Prompt:"
    ),
    "q_target_p": (
       "### Query:{q} ### Response:{target} ### Prompt:"
    ),
    "q_p": (
       "### Query:{q} ### Prompt:"
    ),
}



def select_gt_zero_reward_indexes(lst,interval=1):
    selected_indexes = []
    for i in range(0, len(lst), interval):
        # 获取当前组的元素和对应索引
        group = lst[i:i+interval]
        group_indexes = list(range(i, min(i+interval, len(lst))))

        # 检查组内是否有大于0的元素
        positive_indexes = [idx for idx, val in zip(group_indexes, group) if val > 0]

        # 如果有大于0的元素，选取第一个；否则，选取组内的第一个元素
        selected_index = positive_indexes[0] if positive_indexes else group_indexes[0]
        selected_indexes.append(selected_index)

    return selected_indexes


def repeat_texts_l(l,repeat_times=1):
    return list(chain.from_iterable(repeat(item, repeat_times) for item in l))




def get_data(data_args):
    if data_args.prompt_type!= "q_p":
        raise NotImplementedError()
    all_queries_path = data_args.all_queries_path
    split_path = data_args.split_path
    with open(all_queries_path) as f:
        all_queries = json.load(f)
    with open(split_path) as f:
        splits = json.load(f)
    list_qs = []
    q_s = splits[data_args.split]
    q_s = sorted(q_s)
    for q in q_s:
        list_qs.append(dict(q = q, target = all_queries[q]))
    print("****************")
    print(list_qs[0])
    print("****************")
    return list_qs



set_seed(42)

def attack_collate_fn(batch):
    collated_batch = {}
    for item in batch:
        for key, value in item.items():
            if key in collated_batch:
                collated_batch[key].append(value)
            else:
                collated_batch[key] = [value]
    return collated_batch  

def set_pad_token(t):
    if t.pad_token is None:
        t.pad_token = t.eos_token
    
    return t


def get_batch(l,bs):
    for i in range(0,len(l),bs):
        yield l[i: i+bs]

def do_reps(
    source_texts, 
    num_reps
):
    source_reps = []
    for text in source_texts: 
        for _ in range(num_reps): 
            source_reps.append(text)
    return source_reps


@hydra.main(config_path="./myconfig", config_name="config_prompter_evaluate")
def main(config: "DictConfig"):

    print(OmegaConf.to_yaml(config), color='red')


    start_time = time.time()
    Path(config.s_p_t_dir).mkdir(exist_ok= True, parents= True)

    config.reward_lm.batch_size = config.batch_size
    config.target_lm.batch_size = config.batch_size

    s_p_t_dir = config.s_p_t_dir
    if config.prompt_way == "prompter":
        promptway_name = config.prompt_way + "_" + config.data_args.prompt_type
        decode_way = f"decode_{config.prompter_lm.generation_configs.name}"
        if config.prompter_lm.generation_configs.num_return_sequences > 1:
            decode_way += f"_numreturn_{config.prompter_lm.generation_configs.num_return_sequences}"
        s_p_t_dir = os.path.join(s_p_t_dir,f"{config.data_args.split}|prompter_{config.prompter_lm.show_name}|{decode_way}|promptway_{promptway_name}")

    s_p_t_dir = os.path.join(s_p_t_dir,f"{config.target_lm.show_name}|max_new_tokens_{config.target_lm.generation_configs.max_new_tokens}")
    Path(s_p_t_dir).mkdir(exist_ok= True, parents= True)

    try:
        save_path = os.path.join(s_p_t_dir,f"targetlm_do_sample_{config.target_lm.generation_configs.do_sample}|append_label_length_{config.append_label_length}.jsonl")
        with open(save_path) as f:
            existed_lines = len(f.readlines())
    except:
        existed_lines = 0
    fp = jsonlines.open(save_path,"a")

    processed_data = get_data(config.data_args)
    processed_data = processed_data[existed_lines:]
    print(len(processed_data),'len(processed_data)')
    if len(processed_data) == 0:
        return

    print(OmegaConf.to_yaml(config), color='red')
    
    target_model_tokenizer = None
    if config.append_label_length != -1:
        # only for selecting tokens at the front
        target_model_tokenizer = set_pad_token(AutoTokenizer.from_pretrained(config.target_lm.model_name,padding_side = "right"))
    reward_lm_fn = create_reward(config)
    target_lm_fn = create_targetlm(config)
    prompter_lm_fn = None
    if config.prompt_way == "prompter":
        prompter_lm_fn = create_prompterlm(config)

    evaluate_fn(target_model_tokenizer,reward_lm_fn,target_lm_fn,prompter_lm_fn,processed_data,config,fp)
    end_time = time.time()

    # 计算运行时间
    elapsed_time = end_time - start_time

    print(f"函数运行时间：{elapsed_time}秒")



@torch.no_grad()
def evaluate_fn(target_model_tokenizer,reward_lm_fn,target_lm_fn,prompter_lm_fn,processed_data,config,fp):
    prompt_template = PROMPT_DICT[config.data_args.prompt_type]
    progress_keys = tqdm(processed_data, total=len(processed_data),desc="keys iteration")
    
    for batch in get_batch(processed_data,config.batch_size):
        batch = attack_collate_fn(batch)
        q_s = batch["q"]
        if config.prompt_way == "prompter":
            # if config.data_args.prompt_type == "q_target_lm_generation_p":
            #     for_prompter_s = batch["target_lm_generation"]
            #     prompter_lm_inputs = [prompt_template.format(q = q_s[index],target_lm_generation = for_prompter_s[index]) for index in range(len(q_s))]
            # elif config.data_args.prompt_type == "q_target_p":
            #     for_prompter_s = batch["target"]
            #     prompter_lm_inputs = [prompt_template.format(q = q_s[index],target = for_prompter_s[index]) for index in range(len(q_s))]
            if config.data_args.prompt_type == "q_p":
                prompter_lm_inputs = [prompt_template.format(q = q_s[index]) for index in range(len(q_s))]
            p_s = prompter_lm_fn(prompter_lm_inputs)
        else:
            raise NotImplementedError()

        assert len(q_s)*config.prompter_lm.generation_configs.num_return_sequences == len(p_s)
        repeat_q_s = repeat_texts_l(q_s,config.prompter_lm.generation_configs.num_return_sequences)
        assert len(repeat_q_s) == len(p_s)
        

        label_s_tokens_decode = None
        # 这里有问题 see   /fs/ess/PAA0201/lzy37ld/why_attack/ckpt/prompter_victim=llama2-7b-chat_prompt_type=q_p_model_name=llama2-base_sample_way_and_n_sample=random_nsample=200_epoch_5/checkpoint-35000-val_analysis
        target_lm_generations = target_lm_fn.get_target_lm_generation(repeat_q_s,p_s,after_sys_tokens = label_s_tokens_decode)
        reward_scores = reward_lm_fn(repeat_q_s,target_lm_generations)
        reward_scores = reward_scores.cpu().tolist()
        gt_zero_reward_index = select_gt_zero_reward_indexes(reward_scores,interval=config.prompter_lm.generation_configs.num_return_sequences)
        selected_rewards = [reward_scores[i] for i in gt_zero_reward_index]
        selected_target_lm_generations = [target_lm_generations[i] for i in gt_zero_reward_index]
        selected_p_s = [p_s[i] for i in gt_zero_reward_index]
        assert len(q_s) == len(selected_rewards) == len(selected_target_lm_generations) == len(selected_p_s)



        p_s = selected_p_s
        target_lm_generations = selected_target_lm_generations
        reward_scores = selected_rewards



        ppl_scores = [-1 for _ in range(len(q_s))]
        if config.ppl:
            ppl_scores = target_lm_fn.ppl_run(q_s,p_s)

        for i in range(len(reward_scores)):
            fp.write(dict(q = q_s[i],p = p_s[i],target_lm_generation = target_lm_generations[i],reward = reward_scores[i],ppl_score = ppl_scores[i],prompter_lm_inputs = prompter_lm_inputs[i]))
        
        progress_keys.update(config.batch_size)



if __name__ == "__main__":
    main()
