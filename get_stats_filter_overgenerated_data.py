
import multiprocessing
import jsonlines
from types import SimpleNamespace
from collections import defaultdict as ddict
import random
random.seed(42)
import json
import os
import pathlib
from utility import deter_if_harm

def process_data(line):
    is_harm = deter_if_harm(harm_scores=[line["reward"]],target_lm_generations=[line["target_lm_generation"]],determine_way = "all")[0]
    return (line["q"],line["p"],line["loss"],line["reward"],line["target_lm_generation"],line["target"],is_harm) 

def read_and_dedup(path):
    datas = []
    unique_lines = set()  # 用于存储唯一行的字典

    with jsonlines.open(path) as f:
        all_lines = list(f)  # 创建一个带有索引的行列表

        with multiprocessing.Pool(10) as pool:
            results = pool.imap_unordered(process_data, all_lines,chunksize=10000)
            for q,p,loss,reward,target_lm_generation,target,is_harm in results:
                if is_harm:
                    if (q,p) not in unique_lines:
                        unique_lines.add((q,p,target_lm_generation))  # 保留具有唯一值的行的索引
                        # 考虑把target也加进去。。。
                        # 带上uuid，来自哪个GCG哪个model，每一个标号，也可以直接分train_test出来，带上train test表示符号，train——test是针对query来说的。。那train test就需要针对来改一改了。。 存为两个文件？
                        datas.append(dict(q = q,p = p, loss = loss, reward = reward, target_lm_generation = target_lm_generation, target = target))
    return datas

def get_q_dict(datas,n_sample):
    q_dict = ddict(list)
    for item in datas:
        assert item["reward"] > 0
        q_dict[item["q"]].append(item)

    return q_dict




path_template = "/fs/ess/PAA0201/lzy37ld/why_attack_lzy/data/s_p_t_evaluate/llama2-7b-chat|max_new_tokens_60/offset_{offset}|promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl"
q_dict_list = []
n_sample = 100
for offset in range(0,520,10):

    path = path_template.format(offset = offset)
    if not os.path.exists(path):
        print(f"{path} not exist")
        continue

    q_dict = get_q_dict(read_and_dedup(path),n_sample = n_sample)
    q_dict_list.append(q_dict)


combined_dict = {key: value for d in q_dict_list for key, value in d.items()}
combined_dict_len = {}
for key in combined_dict:
    combined_dict_len[key] = len(combined_dict[key])

save_dir = "./stats_s_p_t_evaluate/llama2-7b-chat|max_new_tokens_60"
pathlib.Path(save_dir).mkdir(exist_ok= True, parents= True)
with open(os.path.join(save_dir,"stats.json"),"w") as f:
    json.dump(combined_dict_len,f)

    
