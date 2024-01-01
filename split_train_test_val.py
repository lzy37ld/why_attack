
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
import pathlib
random.seed(42)


@hydra.main(config_path="./myconfig", config_name="config_split")
def main(config: "DictConfig"):
    with open(config.all_queries_path) as f:
        all_queries = list(json.load(f).keys())
    with open(config.success_jb_path) as f:
        all_jb_queries = list(json.load(f).keys())
    train_queries = random.sample(all_jb_queries,config.num_train_queries)
    assert(config.num_train_queries == len(all_jb_queries))
    rest_queries = list(set(all_queries) - set(train_queries))
    test_queries = random.sample(rest_queries,config.num_test_queries)
    val_queries = list(set(rest_queries) - set(test_queries))
    d = {}
    d["train"] = train_queries
    d["test"] = test_queries
    d["val"] = val_queries

    with open(config.all_checked_q_s) as f:
        all_checked_q_s = json.load(f)["all_checked_queries"]
    d["hard"]=list(set(all_checked_q_s) - set(train_queries))
    d["unknown"]=list(set(all_queries) - set(train_queries) -set(d["hard"]))
    pathlib.Path(config.save_dir).mkdir(exist_ok = True, parents = True)
    save_path = os.path.join(config.save_dir,"train_val_test.json")
    with open(save_path,"w") as f:
        json.dump(d,f)


if __name__ == "__main__":
    main()