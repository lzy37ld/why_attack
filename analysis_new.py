import json
import jsonlines
from argparse import ArgumentParser
from collections import defaultdict as ddict
import os
from pathlib import Path
from utility import deter_if_harm_new
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
	
	if os.path.exists(save_path) and not args.print_only:
		return 1

	q_s_harm = ddict( lambda: ddict(list))
	q_s = []
	target_lm_generations = []
	p_s = []
	beaver_scores = []
	harmbench_scores = []
	gpt_scores = []
	mean_ppl = -1
	avg_suffix_jbs = -1
	with jsonlines.open(args.path) as f:
		for line in f:
			q_s.append(line["q"])
			beaver_scores.append(line["reward"])
			harmbench_scores.append(line["harmbench_reward"])
			ppl_value = line["ppl_q_p"]
			target_lm_generations.append(line["target_lm_generation"])
			q_s_harm[line["q"]]["beaver_scores"].append(line["reward"])
			q_s_harm[line["q"]]["harmbench_scores"].append(line["harmbench_reward"])
			q_s_harm[line["q"]]["target_lm_generations"].append(line["target_lm_generation"])
			q_s_harm[line["q"]]["p"].append(line["p"])

			if args.include_gptscore:
				gpt_scores.append(line["gpt4_reward"])
				q_s_harm[line["q"]]["gpt_scores"].append(line["gpt4_reward"])

			if ppl_value is not None:
				q_s_harm[line["q"]]["ppl"].append(ppl_value)


	if not args.include_gptscore:
		all_harms = deter_if_harm_new(beaver_scores,target_lm_generations,harmbench_scores,determine_way=args.determine_way)
	else:
		all_harms = deter_if_harm_new(beaver_scores,target_lm_generations,harmbench_scores,gpt_scores = gpt_scores,determine_way=args.determine_way)


	print('len(all_harms)',len(all_harms))
	print(len([1 for _ in harmbench_scores if _ == "Yes"]))
	
	all_harms_over_qs = []
	all_num_jbs_over_jbqs = []
	q_ppl = []
	print('len(q_s_harm)',len(q_s_harm))
	for q in q_s_harm:
		

		if not args.include_gptscore:
			q_harms = deter_if_harm_new(q_s_harm[q]["beaver_scores"],q_s_harm[q]["target_lm_generations"],q_s_harm[q]["harmbench_scores"],determine_way=args.determine_way)
		else:
			q_harms = deter_if_harm_new(q_s_harm[q]["beaver_scores"],q_s_harm[q]["target_lm_generations"],q_s_harm[q]["harmbench_scores"],gpt_scores = q_s_harm[q]["gpt_scores"],determine_way=args.determine_way)

		if any(q_harms):
			all_harms_over_qs.append(1)
			num_of_jb_p_s = [q_s_harm[q]["p"][index] for index,_ in enumerate(q_harms) if _ > 0]
			# print(num_of_jb_p_s)
			all_num_jbs_over_jbqs.append(len(set(num_of_jb_p_s)))
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
	parser.add_argument("--include_gptscore",action='store_true')
	parser.add_argument("--print_only",action="store_true")
	args = parser.parse_args()
	main(args)