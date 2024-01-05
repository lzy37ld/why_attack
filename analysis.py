import json
import jsonlines
from argparse import ArgumentParser
from collections import defaultdict as ddict
import os
from pathlib import Path
from utility import deter_if_harm
import numpy as np



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
				ppl_scores.append(line["ppl_score"])
			except:
				ppl_scores.append(line["ppl_q_p"])
			target_lm_generations.append(line["target_lm_generation"])
			q_s_harm[line["q"]]["harm_scores"].append(line["reward"])
			q_s_harm[line["q"]]["target_lm_generations"].append(line["target_lm_generation"])

	all_harms = deter_if_harm(harm_scores,target_lm_generations,determine_way=args.determine_way)
	print(len(all_harms))
	ppl_mean = sum(a * b for a, b in zip(all_harms, ppl_scores))/sum(all_harms)
	
	all_harms_over_qs = []
	print('len(q_s_harm)',len(q_s_harm))
	for q in q_s_harm:
		# print(len(q_s_harm[q]["harm_scores"]))
		q_harms = deter_if_harm(q_s_harm[q]["harm_scores"],q_s_harm[q]["target_lm_generations"],determine_way=args.determine_way)
		if any(q_harms):
			all_harms_over_qs.append(1)
		else:
			# print(q)
			all_harms_over_qs.append(0)

	print("asr_over_all_instances",sum(all_harms)/len(all_harms))
	print("asr_over_all_qs",sum(all_harms_over_qs)/len(all_harms_over_qs))
	print("ppl_mean",ppl_mean)
	if not args.print_only:
		Path(args.save_dir).mkdir(exist_ok= True, parents= True)
		with open(os.path.join(args.save_dir,f"{args.determine_way}|{args.path.split('/')[-1]}"),"w") as f:
			json.dump(dict(
						asr_over_all_instances = round(sum(all_harms)/len(all_harms),2),
						asr_over_all_qs = round(sum(all_harms_over_qs)/len(all_harms_over_qs),2),
						ppl_mean = ppl_mean
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