import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from reward_target_lm import create_reward,create_targetlm
from torch.utils.data import DataLoader
import jsonlines
import os
from tqdm import tqdm
from pathlib import Path
from accelerate.utils import set_seed
from print_color import print

set_seed(42)
def check_torch_dtype(config):
    kwargs = {}
    if config.torch_dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    return kwargs


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


@hydra.main(config_path="./myconfig", config_name="config_evaluate")
def main(config: "DictConfig"):
    datas = []
    device = "cuda"

    with jsonlines.open(config.data) as reader:
        for line in reader:
            datas.append(line)
    print(OmegaConf.to_yaml(config), color='red')
    
    t = AutoTokenizer.from_pretrained(config.target_lm.model_name,padding_side = "right")
    kwargs = check_torch_dtype(config.target_lm)
    model = AutoModelForCausalLM.from_pretrained(config.target_lm.model_name, **kwargs,device_map = "auto")
    

if __name__ == "__main__":
    main()
