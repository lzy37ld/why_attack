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

python evaluate.py prompt_way=own append_label_length=-1 target_lm.generation_configs.do_sample=True
python evaluate.py prompt_way=own append_label_length=3 target_lm.generation_configs.do_sample=True
python evaluate.py prompt_way=own append_label_length=6 target_lm.generation_configs.do_sample=True
