#!/bin/bash
set -x 
set -e

export offset=$1

python evaluate_for_each_instances.py prompt_way=own batch_size=48 target_lm=llama2 offset=$offset