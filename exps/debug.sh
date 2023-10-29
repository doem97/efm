#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

python  -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env \
    --master_port 40000 \
    ./336_p84_4.py \
    | tee -a ./outputs/debug.txt

