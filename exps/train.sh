#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export THREADS=4

python  -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    --master_port 40000 \
    ./336_p84_4.py \
    | tee -a ./outputs/in1k_336_84_4/log.txt

