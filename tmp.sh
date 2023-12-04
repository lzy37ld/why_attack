#!/bin/bash
set -x 
set -e



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
