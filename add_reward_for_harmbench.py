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
import random
import jsonlines
set_seed(42)


fail_reward = {
    "reward": -100,
    "harmbench_reward": "No"
}



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


@hydra.main(config_path="./myconfig", config_name="config_reward_harmbench")
def main(config: "DictConfig"):
    config.reward_lm.batch_size = config.batch_size
    print(OmegaConf.to_yaml(config), color='red')

    
    path = config.path
    datas = []
    if "beaver" in config.reward_lm.model_name.lower():
        reward_name = "reward"
    elif "harmbench" in config.reward_lm.model_name.lower():
        reward_name = "harmbench_reward"
    else:
        raise NotImplementedError()
    
    with jsonlines.open(path) as f:
        for line in f:
            datas.append(line)

    if reward_name in datas[0].keys():
        print("The reward already existed in the file")
        exit(0)

    reward_lm_fn = create_reward(config)

    only_prev_harm_datas_indexes = list(range(len(datas)))
    if config.only_prev_harm:
        assert "reward" in datas[0].keys()
        only_prev_harm_datas_indexes = [index for index,_ in enumerate(datas) if _["reward"] > 0]

    selected_datas = [datas[index] for index in only_prev_harm_datas_indexes]
    print('len(selected_datas)',len(selected_datas))
    # print('selected_datas',selected_datas)

    rewards = evaluate_fn(reward_lm_fn,selected_datas,config)
    assert len(rewards) == len(selected_datas)
    print(rewards)

    save_rewards = rewards

    left_indexes = set(range(len(datas))) - set(only_prev_harm_datas_indexes)
    print("len(left_indexes)",len(left_indexes))
    if len(left_indexes) > 0:
        integrated_rewards = [fail_reward[reward_name]] * len(datas)
        for reward, index in zip(rewards,only_prev_harm_datas_indexes):
            integrated_rewards[index] = reward
        save_rewards = integrated_rewards

    assert len(save_rewards) == len(datas)
    for index,data in enumerate(datas):
        data.update({reward_name:save_rewards[index]})
    
    with jsonlines.open(path,"w",flush= True) as f:
        f.write_all(datas)



@torch.no_grad()
def evaluate_fn(reward_lm_fn,datas,config):
    all_rewards = []
    progress = tqdm(datas,total=len(datas),desc="# of data")

    for batch in get_batch(datas,config.batch_size):
        q_s = [_["q"] for _ in batch]
        # p_s = [_["p"] for _ in batch]
        repeat_for_targetlm_q_s = [q_s[index] for index in range(len(q_s))]
        target_lm_generations = [_["target_lm_generation"] for _ in batch]
        
        reward_scores = reward_lm_fn(repeat_for_targetlm_q_s,target_lm_generations)
        all_rewards.extend(reward_scores)
        progress.update(len(batch))

    return all_rewards


if __name__ == "__main__":
    main()

