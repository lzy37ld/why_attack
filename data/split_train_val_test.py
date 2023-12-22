import csv
import os
import random
from collections import defaultdict as ddict
import json

random.seed(42)


# existing_files_dir = '/home/liao.629/why_attack/s_p_t_evaluate/vicuna-7b-chat-v1.5|max_new_tokens_60'
# available_offsets = list(os.listdir(existing_files_dir))
# remove="|promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl"
# available_offsets = [_.replace(remove,"") for _ in available_offsets]
# print(len(available_offsets))


# base on   /fs/ess/PAA0201/lzy37ld/why_attack/data/results_n_steps_1000_llama2-chat
# train_offsets based on llama2-chat results_n_steps

vicuna 的train setting对不上train offsets，需要重新来一遍。
available_offsets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 160, 170, 180, 190, 200, 210, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 360, 370, 380, 390, 400, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510]
available_offsets = [f"offset_{_}" for _ in available_offsets]
interval=10
total_offset = 52
# total 520
num_train_offsets = int(400/interval)
num_val_offsets = int(40/interval)
num_test_offsets = int(80/interval)
# train_offsets = random.sample(available_offsets,num_train_offsets)
train_offsets = available_offsets[:num_train_offsets]

path = "/home/liao.629/why_attack/data/harmful_behaviors.csv"
offset_q_target_d = ddict(dict)
with open(path) as f:
	reader =csv.reader(f)
	next(reader)
	tmp_d = {}
	offset_count = 0
	for i,line in enumerate(reader):
		index = i + 1
		tmp_d[line[0]] = line[1]
		if index % interval == 0:
			assert len(tmp_d) == interval
			offset_q_target_d[f"offset_{offset_count}"] = tmp_d
			offset_count += interval
			tmp_d = {}
val_offsets = random.sample(list(set(offset_q_target_d.keys()) - set(train_offsets)),num_val_offsets)
test_offsets = list(set(offset_q_target_d.keys()) - set(train_offsets) - set(val_offsets))


print(sorted([int(_.replace("offset_","")) for _ in train_offsets]))
print(len([int(_.replace("offset_","")) for _ in train_offsets]))



save_splits = {"train":train_offsets,"val":val_offsets,"test":test_offsets}
with open("/home/liao.629/why_attack/data/splits.json","w") as f:
	json.dump(save_splits,f)



with open("/home/liao.629/why_attack/data/offset_q_target.json","w") as f:
	json.dump(offset_q_target_d,f)


