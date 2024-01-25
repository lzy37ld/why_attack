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
	q_s_harm = ddict( lambda: ddict(list))
	q_s = []
	target_lm_generations = []
	p_s = []
	harm_scores = []
	ppl_scores = []
	with jsonlines.open(args.path) as f:
		for line in f:
			q_s.append(line["q"])
			harm_scores.append(line["reward"])
			try:
				ppl_value = line["ppl_score"]
			except:
				ppl_value = line["ppl_q_p"]
				
			target_lm_generations.append(line["target_lm_generation"])
			q_s_harm[line["q"]]["harm_scores"].append(line["reward"])
			q_s_harm[line["q"]]["target_lm_generations"].append(line["target_lm_generation"])
			if ppl_value is not None:
				q_s_harm[line["q"]]["ppl"].append(ppl_value)
				ppl_scores.append(ppl_value)

	all_harms = deter_if_harm(harm_scores,target_lm_generations,determine_way=args.determine_way)
	print('len(all_harms)',len(all_harms))
	
	all_harms_over_qs = []
	q_ppl = []
	print('len(q_s_harm)',len(q_s_harm))
	for q in q_s_harm:
		# print(len(q_s_harm[q]["harm_scores"]))
		q_harms = deter_if_harm(q_s_harm[q]["harm_scores"],q_s_harm[q]["target_lm_generations"],determine_way=args.determine_way)
		if any(q_harms):
			all_harms_over_qs.append(1)
		else:
			# print(q)
			all_harms_over_qs.append(0)
		if "ppl" in q_s_harm[q]:
			q_ppl.append(find_min_corresponding_b(q_harms,q_s_harm[q]["ppl"]))

	print("asr_over_all_instances",sum(all_harms)/len(all_harms))
	print("asr_over_all_qs",sum(all_harms_over_qs)/len(all_harms_over_qs))
	# print(q_ppl)
	print('sum(all_harms_over_qs)',sum(all_harms_over_qs))


	q_ppl_wo_zero = [_ for _ in q_ppl if _ != 0 ]

	with open(args.base_ppl) as f:
		base_ppl = json.load(f)
	base_ppl_values = list(sorted(list(base_ppl.keys()),reverse= True))
	for index,threshold_ppl in enumerate(base_ppl_values):
		# when > (instead of greater or equal) than threshold, then take it as the one filtered by ppl detecotr
		False_Positive = round(index / len(base_ppl_values),2)
		ASR = len([_ for _ in q_ppl_wo_zero if _ <= threshold_ppl]) / len(q_ppl)
		if index == len(base_ppl_values)-1:
			print(threshold_ppl,'threshold_ppl')






if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--path",default="/home/liao.629/why_attack/s_p_t_evaluate/promptway_no|targetlm_do_sample_False|append_label_length_-1.jsonl")
	parser.add_argument("--save_dir",default="./analysis_ASR")
	parser.add_argument("--determine_way",choices=["all","score","em"],default="all")
	parser.add_argument("--base_ppl",default="/users/PAA0201/lzy37ld/why_attack_lzy/data/q_ppl/llama2-7b-chat/q_ppl_d.json")
	args = parser.parse_args()
	print("*"*50)
	print("THis is your base ppl files, pls decide it's llama2 or vicuna")
	print(args.base_ppl)
	print("*"*50)
	main(args)