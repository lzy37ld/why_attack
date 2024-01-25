import json
import jsonlines
from argparse import ArgumentParser
from collections import defaultdict as ddict
import os
from pathlib import Path
from utility import deter_if_harm
import numpy as np


def find_min_corresponding_b(a, b):
    """
    Function to find the minimum value in 'b' at the positions where 'a' has 1.
    If 'a' has no 1s, it returns -1.
    """
    min_val = float('inf')
    found = False

    for i in range(len(a)):
        if a[i] == 1:
            found = True
            if b[i] < min_val:
                min_val = b[i]

    return min_val if found else 0

def main(args):
	Path(args.save_dir).mkdir(exist_ok= True, parents= True)
	save_path = os.path.join(args.save_dir,f"{args.determine_way}|{args.path.split('/')[-1]}")
	if os.path.exists(save_path):
		return 1

	q_s_harm = ddict( lambda: ddict(list))
	q_s = []
	target_lm_generations = []
	p_s = []
	harm_scores = []
	mean_ppl = -1
	avg_suffix_jbs = -1
	with jsonlines.open(args.path) as f:
		for line in f:
			q_s.append(line["q"])
			harm_scores.append(line["reward"])
			ppl_value = line["ppl_q_p"]
			target_lm_generations.append(line["target_lm_generation"])
			q_s_harm[line["q"]]["harm_scores"].append(line["reward"])
			q_s_harm[line["q"]]["target_lm_generations"].append(line["target_lm_generation"])
			if ppl_value is not None:
				q_s_harm[line["q"]]["ppl"].append(ppl_value)

	all_harms = deter_if_harm(harm_scores,target_lm_generations,determine_way=args.determine_way)
	print('len(all_harms)',len(all_harms))
	
	all_harms_over_qs = []
	all_num_jbs_over_jbqs = []
	q_ppl = []
	print('len(q_s_harm)',len(q_s_harm))
	for q in q_s_harm:
		q_harms = deter_if_harm(q_s_harm[q]["harm_scores"],q_s_harm[q]["target_lm_generations"],determine_way=args.determine_way)
		if any(q_harms):
			all_harms_over_qs.append(1)
			all_num_jbs_over_jbqs.append(sum(q_harms))
		else:
			all_harms_over_qs.append(0)
		if "ppl" in q_s_harm[q]:
			q_ppl.append(find_min_corresponding_b(q_harms,q_s_harm[q]["ppl"]))

	print("asr_over_all_instances",sum(all_harms)/len(all_harms))
	
	print("asr_over_all_qs",sum(all_harms_over_qs)/len(all_harms_over_qs))
	print('sum(all_harms_over_qs)',sum(all_harms_over_qs))

	if sum(all_harms_over_qs) != 0:
		mean_ppl = sum(q_ppl)/sum(all_harms_over_qs)

	print("ppl average over one instance for each query if it exist one instance to Jailbreak.",mean_ppl)

	# only consider the queries having at least >= 1 jbs
	if len(all_num_jbs_over_jbqs) != 0:
		avg_suffix_jbs = sum(all_num_jbs_over_jbqs)/len(all_num_jbs_over_jbqs)
	print("#jbs for queries. only consider the queries having at least >= 1 jbs",avg_suffix_jbs)
	
	if not args.print_only:
		with open(save_path,"w") as f:
			json.dump(dict(
						asr_over_all_instances = round(sum(all_harms)/len(all_harms),2),
						asr_over_all_qs = round(sum(all_harms_over_qs)/len(all_harms_over_qs),2),
						mean_ppl = round(mean_ppl,2),
						avg_suffix_jbs = round(avg_suffix_jbs,2)
						),
						f)



if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--path",default="/home/liao.629/why_attack/s_p_t_evaluate/promptway_no|targetlm_do_sample_False|append_label_length_-1.jsonl")
	parser.add_argument("--save_dir",default="./analysis")
	parser.add_argument("--determine_way",choices=["all","score","em"],default="all")
	parser.add_argument("--print_only",action="store_true")
	args = parser.parse_args()
	main(args)