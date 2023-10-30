#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export THREADS=4
# export PATCHES=6
# export NUM_PAIRS=1
export PATCHES=3
export NUM_PAIRS=4

# export PATCHES=2
# export NUM_PAIRS=2

python  -m torch.distributed.launch \
    --nproc_per_node=$THREADS \
    --use_env \
    --master_port 40000 \
    ./trainperm.py \
    --epochs 100 \
    --patches $PATCHES \
    --num_pairs $NUM_PAIRS \
    | tee -a ./outputs/336_${PATCHES}x${PATCHES}_${NUM_PAIRS}.log