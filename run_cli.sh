#!/bin/bash
set -x 
set -e


# multi_0
prompway = no
python evaluate.py prompt_way=no append_label_length=-1
python evaluate.py prompt_way=no append_label_length=3
python evaluate.py prompt_way=no append_label_length=6

python evaluate.py prompt_way=no append_label_length=-1 target_lm.generation_configs.do_sample=True
python evaluate.py prompt_way=no append_label_length=3 target_lm.generation_configs.do_sample=True
python evaluate.py prompt_way=no append_label_length=6 target_lm.generation_configs.do_sample=True


# promptway = own
python evaluate.py prompt_way=own append_label_length=-1
python evaluate.py prompt_way=own append_label_length=3
python evaluate.py prompt_way=own append_label_length=6

python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.do_sample=True
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.do_sample=True
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.do_sample=True





# prompway = no
python evaluate.py prompt_way=no append_label_length=-1 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=no append_label_length=3 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=no append_label_length=6 target_lm.generation_configs.max_new_tokens=100

python evaluate.py prompt_way=no append_label_length=-1 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=no append_label_length=3 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=no append_label_length=6 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100


# promptway = own
python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.max_new_tokens=100

python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100






# multi_1
# max_new_tokens = 60

# promptway = own
python evaluate.py prompt_way=own append_label_length=-1 data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=3 data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=6 data=./data/mutli_1_reformatted_attack.jsonl multi=1

python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.do_sample=True data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.do_sample=True data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.do_sample=True data=./data/mutli_1_reformatted_attack.jsonl multi=1




# max_new_tokens = 100

# promptway = own
python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.max_new_tokens=100 data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.max_new_tokens=100 data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.max_new_tokens=100 data=./data/mutli_1_reformatted_attack.jsonl multi=1

python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100 data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100 data=./data/mutli_1_reformatted_attack.jsonl multi=1
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100 data=./data/mutli_1_reformatted_attack.jsonl multi=1




# evaluate_for_test.py
python evaluate_for_each_instances.py prompt_way=own batch_size=48 target_lm=llama2-chat offset=?
python evaluate_for_each_instances.py prompt_way=own batch_size=48 offset=0 data_dir=results_n_steps_500_vicuna data_prefix="individual_behaviors_vicuna_gcg_offset\{offset\}.json" target_lm=vicuna-chat





python check_tokens.py data="'/home/liao.629/why_attack/s_p_t_evaluate/promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl'"
python check_tokens.py data="'/home/liao.629/why_attack/s_p_t_evaluate/promptway_no|targetlm_do_sample_False|append_label_length_-1.jsonl'"
