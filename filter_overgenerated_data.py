
import multiprocessing
import jsonlines
from types import SimpleNamespace
from collections import defaultdict as ddict
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import copy
import json
import os
from tqdm import tqdm
import time
import pathlib
random.seed(42)
from utility import deter_if_harm



def get_d_total_len(d):
    total_len = 0
    for key in d:
        total_len += len(d[key])
    return total_len

def agg_d(d):
    l = []
    for key in d:
        l.extend(d[key])
    return l

def sample_iteratively_by_key(d, n_sample):
    l = []
    keys = list(sorted(list(d.keys()), key= lambda x: int(x.split("_")[1])))
    break_while = False
    while True:
        _keys = copy.deepcopy(keys)
        random.shuffle(_keys)
        for key in _keys:
            if len(d[key]) == 0:
                continue
            sample = random.sample(d[key],1)[0]
            d[key].remove(sample)
            l.append(sample)
            if len(l) == n_sample:
                break_while = True
                break
        if break_while:
            break
    return l

def process_data(line,determine_way):
    is_harm = deter_if_harm(harm_scores=[line["reward"]],target_lm_generations=[line["target_lm_generation"]],determine_way = determine_way)[0]
    return (line["q"],line["p"],line["loss"],line["reward"],line["target_lm_generation"],line["target"],line["step"],is_harm) 

def read_and_dedup(path,config):
    datas = []
    unique_lines = set()  # 用于存储唯一行的字典
    _process_data = partial(process_data,determine_way = config.determine_way)
    print("*"*50)
    print(path)
    # with open(path,'r') as f:
    #     for i,line in enumerate(f.readlines()):
    #         try:
    #             data = json.loads(line)
    #             print(data)
    #         except:
    #             print(i)
    #             print(line)
    #             print(i)
    #             exit(1)
    # return -1

    with jsonlines.open(path) as f:
        all_lines = list(f)  # 创建一个带有索引的行列表
    print(f"if the query dont have {config.interval} instances, then drop")
    interval=config.interval
    for m in range(10,0,-1):
        try:
            _ = all_lines[m*interval-1]
            all_lines = all_lines[:m*interval]
            # if one query dont have config.interval instances, then we dont take it for later consideration...
            print(f"keep only {m} queries for later check")
            break
        except:
            continue

    with multiprocessing.Pool(10) as pool:
        results = pool.imap_unordered(_process_data, all_lines,chunksize=10000)
        for q,p,loss,reward,target_lm_generation,target,step,is_harm in results:
            if is_harm:
                if (q,p) not in unique_lines:
                    unique_lines.add((q,p,target_lm_generation))  # 保留具有唯一值的行的索引
                    # 考虑把target也加进去。。。
                    datas.append(dict(q = q,p = p, loss = loss, reward = reward, target_lm_generation = target_lm_generation, target = target,step = step))

    return datas, set([_["q"] for _ in all_lines])

def get_q_dict(datas,config):

    n_sample = config.n_sample
    sample_way = config.sample_way
    return_d = ddict(list)
    if "low_ppl" in sample_way:
        raise NotImplementedError()
    
    if sample_way.startswith("random"):

        q_dict = ddict(list)
        for item in datas:
            assert item["reward"] > 0
            q_dict[item["q"]].append(item)
        for q in q_dict:
            if n_sample <= len(q_dict[q]):
                return_d[q] = random.sample(q_dict[q],n_sample)
            else:
                print("sample_way",sample_way)
                print(f"n_sample = {n_sample}, but {q} only has {len(q_dict[q])} options")
                return_d[q] = q_dict[q]

    elif sample_way.startswith("step"):
            
        q_dict = ddict( lambda : ddict(list))
        
        for item in datas:
            assert item["reward"] > 0
            q_dict[item["q"]][item["step"]].append(item)
        for q in q_dict:
            if n_sample > get_d_total_len(q_dict[q]):
                print("sample_way",sample_way)
                print(f"n_sample = {n_sample}, but {q} only has {get_d_total_len(q_dict[q])} options")
                return_d[q] = agg_d(q_dict[q])
            else:
                return_d[q] = sample_iteratively_by_key(q_dict[q],n_sample)

    elif sample_way.startswith("loss"):
        def get_loss_step_q_dict(sample_way,datas):
            assert len(sample_way.split("_")) == 2, "sample way name is wrong"
            name,loss_num_parts = sample_way.split("_")
            num_parts = int(loss_num_parts)

            def diverders_for_l(lst,num_parts):
                lst.sort()
                n = len(lst)
                dividers = []
                for i in range(1, int(num_parts)):
                    # 计算每个分区的理想分界点
                    divider_index = int(n * i / num_parts)
                    dividers.append(lst[divider_index])
                return dividers
            all_loss = [_["loss"] for _ in datas]
            loss_diverders = sorted(diverders_for_l(all_loss,num_parts))

            q_dict = ddict(lambda : ddict(list))
            for item in datas:
                assert item["reward"] > 0
                loss = item["loss"]
                loss_step = -1
                for loss_range in range(len(loss_diverders)):
                    if loss <= loss_diverders[loss_range]:
                        loss_step = f"loss_{loss_range}"
                        break
                if loss_step == -1:
                    loss_step = f"loss_{len(loss_diverders)}"
                
                q_dict[item["q"]][loss_step].append(item)

            return q_dict
        
        q_dict = get_loss_step_q_dict(sample_way,datas)
        for q in q_dict:
            if n_sample > get_d_total_len(q_dict[q]):
                print("sample_way",sample_way)
                print(f"n_sample = {n_sample}, but {q} only has {get_d_total_len(q_dict[q])} options")
                return_d[q] = agg_d(q_dict[q])
            else:
                return_d[q] = sample_iteratively_by_key(q_dict[q],n_sample)

    else:
        raise NotImplementedError()
    return return_d


@hydra.main(config_path="./myconfig", config_name="config_filter_overgenerated_data")
def main(config: "DictConfig"):
    # "/home/liao.629/why_attack/s_p_t_evaluate/vicuna-7b-chat-v1.5|max_new_tokens_60/\{offset\}|promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl"
    evaluated_data_path_template = config.evaluated_data_path_template

    # train_offsets = ["offset_10", "offset_20", "offset_30", "offset_40", "offset_50", "offset_60", "offset_70", "offset_80", "offset_90", "offset_100"]
    train_offsets = [f'offset_{_}' for _ in range(0,520,10)]
    q_dict_list = []
    
    queries_with_jb = []
    num_all_queries = 0
    all_checked_queries = []
    for offset in tqdm(train_offsets):
        path = evaluated_data_path_template.format(offset = offset)
        if os.path.exists(path):
            with open(path) as f:
                if len(f.readlines()) <=0:
                    print(path,"do not have values")
                    continue
            unfilter_data,checked_queries = read_and_dedup(path,config)
            num_all_queries += len(checked_queries)
            queries_with_jb.extend(list(set([_["q"] for _ in unfilter_data])))
            all_checked_queries.extend(list(checked_queries))

            q_dict = get_q_dict(unfilter_data,config)
            q_dict_list.append(q_dict)
    
    combined_dict = {key: value for d in q_dict_list for key, value in d.items()}
    pathlib.Path(config.save_dir).mkdir(exist_ok= True, parents= True)
    save_path = os.path.join(config.save_dir,f"success_JB_victimmodel={config.evaluated_model}_sampleway={config.sample_way}_nsample={config.n_sample}.json")
    with open(save_path,"w") as f:
        json.dump(combined_dict,f)

    save_path = os.path.join(config.save_dir,f"allchecked_JB_victimmodel={config.evaluated_model}.json")
    with open(save_path,"w") as f:
        json.dump({"all_checked_queries": all_checked_queries},f)


    info_path = os.path.join(config.save_dir,"info.txt")
    nl = "\n"
    
    with open(info_path,"a") as f:
        f.write(nl)
        f.write("*"*50)
        f.write("*"*50)
        f.write("*"*50)
        f.write(nl)
        f.write(f"success_JB_victimmodel={config.evaluated_model}_sampleway={config.sample_way}_nsample={config.n_sample}.json")
        f.write(nl)
        current_timestamp = time.time()
        local_time = time.localtime(current_timestamp)
        f.write(f"time: {time.strftime('%Y-%m-%d %H:%M:%S', local_time)}")
        f.write(nl)
        f.write(nl)
        f.write(f"total #jb queries: {len(queries_with_jb)}")
        f.write(nl)
        f.write(f"total raio jb queries: {len(queries_with_jb)/num_all_queries}")
        f.write(nl)

    



if __name__ == "__main__":
    main()