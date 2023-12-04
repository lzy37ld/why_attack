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
    config.reward_lm.batch_size = config.batch_size
    config.target_lm.batch_size = config.batch_size
    Path(config.s_p_t_dir).mkdir(exist_ok= True, parents= True)

    s_p_t_dir = config.s_p_t_dir
    s_p_t_dir = os.path.join(s_p_t_dir,f"max_new_tokens_{config.target_lm.generation_configs.max_new_tokens}")
    Path(s_p_t_dir).mkdir(exist_ok= True, parents= True)

    path = os.path.join(s_p_t_dir,f"promptway_{config.prompt_way}|targetlm_do_sample_{config.target_lm.generation_configs.do_sample}|append_label_length_{config.append_label_length}.jsonl")


    fp = jsonlines.open(path,"a")
    with open(path) as f:
        existed_lines = len(f.readlines())
    assert existed_lines == 0, "delete it"

    all_unique_qs_datas = []
    with jsonlines.open(config.data) as reader:
        for line in reader:
            all_unique_qs_datas.append(line)
    print(OmegaConf.to_yaml(config), color='red')
    
    target_model_tokenizer = None
    if config.append_label_length != -1:
        # only for selecting tokens at the front
        target_model_tokenizer = set_pad_token(AutoTokenizer.from_pretrained(config.target_lm.model_name,padding_side = "right"))
    reward_lm_fn = create_reward(config)
    target_lm_fn = create_targetlm(config)
    evaluate_fn(target_model_tokenizer,reward_lm_fn,target_lm_fn,all_unique_qs_datas,config,fp)



@torch.no_grad()
def evaluate_fn(target_model_tokenizer,reward_lm_fn,target_lm_fn,all_unique_qs_datas,config,fp):

    with tqdm(total=len(all_unique_qs_datas)) as progress:
            
        for batch in get_batch(all_unique_qs_datas,config.batch_size):
            batch = attack_collate_fn(batch)
            label_s = batch["target"]
            file_s = batch["file"]
            loss_s = batch["loss"]
            q_s = batch["q"]
            if config.prompt_way == "own":
                p_s = batch["p"]
            elif config.prompt_way == "no":
                p_s = batch["p"]
                p_s = ["" for _ in batch["p"]]

            else:
                raise NotImplementedError()

            assert len(q_s) == len(p_s)

            label_s_tokens_decode = None
            if config.append_label_length != -1 and config.prompt_way != "model":
                label_s_tokens = target_model_tokenizer(label_s,padding = True,return_tensors = "pt",add_special_tokens = False).input_ids[:,:config.append_label_length]
                label_s_tokens_decode = target_model_tokenizer.batch_decode(label_s_tokens,skip_special_tokens = True)

            target_lm_generations = target_lm_fn(q_s,p_s,after_sys_tokens = label_s_tokens_decode)
            reward_scores = reward_lm_fn(q_s,target_lm_generations)
            reward_scores = reward_scores.cpu().tolist()

            for i in range(len(reward_scores)):
                fp.write(dict(q = q_s[i],p = p_s[i],target_lm_generation = target_lm_generations[i],reward = reward_scores[i],loss = loss_s[i],target = label_s[i],file = file_s[i]))
            
            progress.update(config.batch_size)


if __name__ == "__main__":
    main()
