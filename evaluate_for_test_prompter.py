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


def get_data(data_args , mode = "train"):

    data_path = data_args.data_path
    prompt_type = data_args.prompt_type
    train_ratio = data_args.train_ratio
    assert 0 < train_ratio < 1,"train ratio >0 <1"
    with open(data_path) as f:
        datas = json.load(f)
    if mode == "train":
        q_s = list(datas.keys())[:int(len(datas) * train_ratio)]
    elif mode == "test":
        q_s = list(datas.keys())[int(len(datas) * train_ratio):]
    else:
        raise NotImplementedError()
    
    list_data_dict = []
    for q in q_s:
        list_data_dict.extend(datas[q])
    
    
    return list_data_dict



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

    s_p_t_dir = config.s_p_t_dir
    if config.prompt_way == "prompter":
        promptway_name = config.prompt_way + "_" + config.data_args.prompt_type
        s_p_t_dir = os.path.join(s_p_t_dir,f"prompter_{config.prompter_lm.show_name}|promptway_{promptway_name}")

    s_p_t_dir = os.path.join(s_p_t_dir,f"{config.target_lm.show_name}|max_new_tokens_{config.target_lm.generation_configs.max_new_tokens}")
    Path(s_p_t_dir).mkdir(exist_ok= True, parents= True)

    try:
        save_path = os.path.join(s_p_t_dir,f"targetlm_do_sample_{config.target_lm.generation_configs.do_sample}|append_label_length_{config.append_label_length}.jsonl")
        with open(save_path) as f:
            existed_lines = len(f.readlines())
        assert existed_lines == 0
    except:
        pass
    fp = jsonlines.open(save_path,"a")

    processed_data = get_data(config.data_args,mode = "test")

    # for target from advbench.
    with jsonlines.open("/home/liao.629/why_attack/data/mutli_0_reformatted_attack.jsonl") as f:
        q_target_d = {}
        for line in f:
            q_target_d[line["q"]] = line["target"]
    
    tmp_processed_data = []
    for _processed_data in processed_data:
        _processed_data["target"] = q_target_d[_processed_data["q"]]
        tmp_processed_data.append(_processed_data)

    # dedup
    processed_data = []
    seen = set()
    for item in tmp_processed_data:
        if config.data_args.prompt_type == "q_target_p":
            identifier = (item['q'], item['target'])
        elif config.data_args.prompt_type == "q_target_lm_generation_p":
            identifier = (item['q'], item['target_lm_generation'])
        else:
            raise NotImplementedError()
        if identifier not in seen:
            seen.add(identifier)
            processed_data.append(item)


    

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

        if config.prompt_way == "own":
            p_s = batch["p"]
        elif config.prompt_way == "no":
            p_s = batch["p"]
            p_s = ["" for _ in batch["p"]]
        elif config.prompt_way == "prompter":
            if config.data_args.prompt_type == "q_target_lm_generation_p":
                for_prompter_s = batch["target_lm_generation"]
                prompter_lm_inputs = [prompt_template.format(q = q_s[index],target_lm_generation = for_prompter_s[index]) for index in range(len(q_s))]
            elif config.data_args.prompt_type == "q_target_p":
                for_prompter_s = batch["target"]
                prompter_lm_inputs = [prompt_template.format(q = q_s[index],target = for_prompter_s[index]) for index in range(len(q_s))]

            p_s = prompter_lm_fn(prompter_lm_inputs)
        else:
            raise NotImplementedError()

        assert len(q_s) == len(p_s)

        label_s_tokens_decode = None
        target_lm_generations = target_lm_fn(q_s,p_s,after_sys_tokens = label_s_tokens_decode)
        reward_scores = reward_lm_fn(q_s,target_lm_generations)
        reward_scores = reward_scores.cpu().tolist()

        for i in range(len(reward_scores)):
            fp.write(dict(q = q_s[i],p = p_s[i],target_lm_generation = target_lm_generations[i],reward = reward_scores[i],for_prompter = for_prompter_s[i]))
        
        progress_keys.update(config.batch_size)



if __name__ == "__main__":
    main()
