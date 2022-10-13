#!/usr/bin/bash

if (( $# < 1 )); then
    >&2 echo "Please specify GPU ID"
    exit 1
fi

echo "Use GPU: $1"

CUDA_VISIBLE_DEVICES=$1 python client.py  --work_dir ./test   --seed -1  --batch_size 32 --num_train_steps 10000 --save_tb --save_buffer --save_video --save_model

