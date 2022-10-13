#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py  --domain_name cheetah  --task_name run  --work_dir ./save_test   --seed -1  --batch_size 32 --num_train_steps 10000  --num_eval_episodes 1  --save_tb --save_buffer --save_video --save_model



