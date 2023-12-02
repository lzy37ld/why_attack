#!/bin/bash
set -x 
set -e


# python evaluate.py prompt_way=no append_label_length=-1 target_lm.generation_configs.max_new_tokens=100
# python evaluate.py prompt_way=no append_label_length=3 target_lm.generation_configs.max_new_tokens=100
# python evaluate.py prompt_way=no append_label_length=6 target_lm.generation_configs.max_new_tokens=100

# python evaluate.py prompt_way=no append_label_length=-1 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
# python evaluate.py prompt_way=no append_label_length=3 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
# python evaluate.py prompt_way=no append_label_length=6 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100


# promptway = own
python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.max_new_tokens=100

python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
