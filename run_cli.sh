#!/bin/bash
set -x 
set -e

# prompway = no
# python evaluate.py prompt_way=no append_label_length=-1
# python evaluate.py prompt_way=no append_label_length=3
# python evaluate.py prompt_way=no append_label_length=6

# python evaluate.py prompt_way=no append_label_length=-1 target_lm.generation_configs.do_sample=True
# python evaluate.py prompt_way=no append_label_length=3 target_lm.generation_configs.do_sample=True
# python evaluate.py prompt_way=no append_label_length=6 target_lm.generation_configs.do_sample=True


# promptway = own
python evaluate.py prompt_way=own append_label_length=-1
python evaluate.py prompt_way=own append_label_length=3
python evaluate.py prompt_way=own append_label_length=6

python evaluate.py prompt_own=own append_label_length=-1 target_lm.generation_configs.do_sample=True
python evaluate.py prompt_own=own append_label_length=3 target_lm.generation_configs.do_sample=True
python evaluate.py prompt_own=own append_label_length=6 target_lm.generation_configs.do_sample=True





# prompway = no
python evaluate.py prompt_way=no append_label_length=-1 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=no append_label_length=3 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=no append_label_length=6 target_lm.generation_configs.max_new_tokens=100

python evaluate.py prompt_way=no append_label_length=-1 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=no append_label_length=3 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_way=no append_label_length=6 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100


# promptway = own
python evaluate.py prompt_own=own append_label_length=-1 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_own=own append_label_length=3 target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_own=own append_label_length=6 target_lm.generation_configs.max_new_tokens=100

python evaluate.py prompt_own=own append_label_length=-1 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_own=own append_label_length=3 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100
python evaluate.py prompt_own=own append_label_length=6 target_lm.generation_configs.do_sample=True target_lm.generation_configs.max_new_tokens=100




python check_tokens.py data="'/home/liao.629/why_attack/s_p_t_evaluate/promptway_own|targetlm_do_sample_False|append_label_length_-1.jsonl'"
python check_tokens.py data="'/home/liao.629/why_attack/s_p_t_evaluate/promptway_no|targetlm_do_sample_False|append_label_length_-1.jsonl'"
