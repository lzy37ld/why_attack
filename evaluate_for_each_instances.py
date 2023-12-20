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
import json
import time

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
    start_time = time.time()
    Path(config.s_p_t_dir).mkdir(exist_ok= True, parents= True)

    config.reward_lm.batch_size = config.batch_size
    config.target_lm.batch_size = config.batch_size

    s_p_t_dir = config.s_p_t_dir
    s_p_t_dir = os.path.join(s_p_t_dir,f"{config.target_lm.show_name}|max_new_tokens_{config.target_lm.generation_configs.max_new_tokens}")
    Path(s_p_t_dir).mkdir(exist_ok= True, parents= True)
    try:
        path = os.path.join(s_p_t_dir,f"offset_{config.offset}|promptway_{config.prompt_way}|targetlm_do_sample_{config.target_lm.generation_configs.do_sample}|append_label_length_{config.append_label_length}.jsonl")
        with open(path) as f:
            existed_lines = len(f.readlines())
        # assert existed_lines == 0
    except:
        existed_lines = 0
    # assert existed_lines == 0, "delete it"
    data_path = os.path.join(config.data_dir,config.data_prefix.format(offset = config.offset))
    # with open(data_path) as f:
    #     lines = len(f.readlines())
    # if lines != 5140113:
    #     print("GCG is still going")
    #     exit(1)


    fp = jsonlines.open(path,"a")


    with open(data_path) as f:
        data = json.load(f)
    goals = data["params"]["goals"]
    targets = data["params"]["targets"]
        
    processed_data = dict()
    only_steps_after = 250
    for index,key in enumerate(goals):
        tmp = {}
        tmp["target"] = targets[index]
        steps_cands = data["steps_cands"][key]
        steps_cands_keys = list(steps_cands.keys())
        for step in steps_cands_keys:
            if int(step.replace("step_","")) < only_steps_after:
                steps_cands.pop(step)
        print(len(steps_cands))
        tmp["steps_cands"] = steps_cands
        processed_data[key] = tmp

    print(OmegaConf.to_yaml(config), color='red')
    
    target_model_tokenizer = None
    if config.append_label_length != -1:
        # only for selecting tokens at the front
        target_model_tokenizer = set_pad_token(AutoTokenizer.from_pretrained(config.target_lm.model_name,padding_side = "right"))

    # process datas, which need to be continued to be executed.
    adv_prompt_per_step_per_instances=config.adv_prompt_per_step_per_instances
    adv_prompt_steps_per_instances=config.adv_prompt_steps_per_instances
    desired_steps_per_instance = (adv_prompt_steps_per_instances-only_steps_after) * adv_prompt_per_step_per_instances
    num_of_covered_instances = existed_lines // desired_steps_per_instance
    num_of_partialy_covered_instances_steps = existed_lines - num_of_covered_instances*desired_steps_per_instance
    num_of_covered_d = dict(num_of_covered_instances=num_of_covered_instances,num_of_partialy_covered_instances_steps=num_of_partialy_covered_instances_steps)

    # reward_lm_fn = None
    # target_lm_fn = None
    reward_lm_fn = create_reward(config)
    target_lm_fn = create_targetlm(config)
    evaluate_fn(target_model_tokenizer,reward_lm_fn,target_lm_fn,processed_data,num_of_covered_d,config,fp)
    end_time = time.time()

    # 计算运行时间
    elapsed_time = end_time - start_time

    print(f"函数运行时间：{elapsed_time}秒")



@torch.no_grad()
def evaluate_fn(target_model_tokenizer,reward_lm_fn,target_lm_fn,processed_data,num_of_covered_d,config,fp):
    progress_keys = tqdm(processed_data, total=len(processed_data),desc="keys iteration")

    for key_index,key in enumerate(processed_data):
        if key_index < num_of_covered_d['num_of_covered_instances']:
            progress_keys.update(1)
            continue

        all_unique_qs_datas = []
        target = processed_data[key]["target"]
        steps_cands = processed_data[key]["steps_cands"]
        for step_index,step in enumerate(steps_cands):
            for item_index,item in enumerate(steps_cands[step]):
                if key_index == num_of_covered_d['num_of_covered_instances'] and step_index * config.adv_prompt_per_step_per_instances + item_index < num_of_covered_d['num_of_partialy_covered_instances_steps']:
                    continue
                tmp = {}
                tmp["target"] = target
                tmp["q"] = key
                tmp["p"] = item["control"]
                tmp["loss"] = item["loss"]
                tmp["step"] = step
                all_unique_qs_datas.append(tmp)


        with tqdm(all_unique_qs_datas, total=len(all_unique_qs_datas),desc="instances iteration",leave= False) as progress_instances:
                
            for batch in get_batch(all_unique_qs_datas,config.batch_size):
                batch = attack_collate_fn(batch)
                label_s = batch["target"]
                # file_s = batch["file"]
                step_s = batch["step"]
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
                    fp.write(dict(q = q_s[i],p = p_s[i],target_lm_generation = target_lm_generations[i],reward = reward_scores[i],loss = loss_s[i],target = label_s[i],step = step_s[i]))
                
                progress_instances.update(config.batch_size)
            progress_keys.update(1)


if __name__ == "__main__":
    main()
