#!/bin/bash
if (( $# < 1 )); then
    >&2 echo "Please specify GPU ID"
    exit 1
fi

echo "Use GPU: $1"

EGL_DEVICE_ID=$1 CUDA_VISIBLE_DEVICES=$1 python3 client.py --num_train_steps 25000 --pre_transform_image_size 116 --work_dir ../main_results/search --batch_size 64 

