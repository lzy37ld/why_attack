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

def select_max_reward_indexes(lst, interval=1):
    selected_indexes = []
    for i in range(0, len(lst), interval):
        # 获取当前组的元素和对应索引
        group = lst[i:i+interval]
        group_indexes = list(range(i, min(i+interval, len(lst))))

        # 找到组内最大元素的索引
        max_value_index = group_indexes[group.index(max(group))]

        selected_indexes.append(max_value_index)

    return selected_indexes

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


    start_time = time.time()
    Path(config.s_p_t_dir).mkdir(exist_ok= True, parents= True)

    config.reward_lm.batch_size = config.batch_size
    config.target_lm.batch_size = config.batch_size
    config.prompter_lm.batch_size = config.batch_size

    s_p_t_dir = config.s_p_t_dir
    s_p_t_dir = os.path.join(s_p_t_dir,f"{config.target_lm.show_name}")
    Path(s_p_t_dir).mkdir(exist_ok= True, parents= True)
    # save_path = os.path.join(s_p_t_dir,f"targetlm_do_sample_{config.target_lm.generation_configs.do_sample}|append_label_length_{config.append_label_length}.jsonl")
    save_path = os.path.join(s_p_t_dir,f"q_ppl_d.json")

    fp = open(save_path,"w")
    processed_data = get_data(config.data_args)

    print(len(processed_data),'len(processed_data)')
    if len(processed_data) == 0:
        return

    print(OmegaConf.to_yaml(config), color='red')
    
    target_model_tokenizer = None

    target_lm_fn = create_targetlm(config)

    evaluate_fn(target_lm_fn,processed_data,config,fp)
    end_time = time.time()

    # 计算运行时间
    elapsed_time = end_time - start_time

    print(f"函数运行时间：{elapsed_time}秒")



@torch.no_grad()
def evaluate_fn(target_lm_fn,processed_data,config,fp):
    progress_keys = tqdm(processed_data, total=len(processed_data),desc="keys iteration")
    q_ppl_d = {}
    for batch in get_batch(processed_data,config.batch_size):
        batch = attack_collate_fn(batch)
        q_s = batch["q"]
        print("*"*50)
        print("This is prompter lm")
        ppl_q_p = [None for _ in range(len(q_s))]
        p_s = [" " for _ in range(len(q_s))]
        if config.ppl:
            print("This is ppl run")
            ppl_q_p = target_lm_fn.ppl_run(q_s,p_s)
            
        
        for i in range(len(q_s)):
            q_ppl_d[q_s[i]] = ppl_q_p[i]
        progress_keys.update(config.batch_size)
    json.dump(q_ppl_d,fp)



if __name__ == "__main__":
    main()
