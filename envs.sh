
EGL_DEVICE_ID=$1 CUDA_VISIBLE_DEVICES=$1 python eval-client.py --domain_name cartpole --task_name swingup --action_repeat 8 --pre_transform_image_size 116 --work_dir ./eval --seed $2 --batch_size 512 --num_train_steps 12500 --save_tb --eval_freq 12500 
EGL_DEVICE_ID=$1 CUDA_VISIBLE_DEVICES=$1 python eval-client.py --domain_name cheetah --task_name run --action_repeat 4 --pre_transform_image_size 116 --work_dir ./eval --seed $2 --batch_size 512 --num_train_steps 25000 --save_tb --eval_freq 12500 
EGL_DEVICE_ID=$1 CUDA_VISIBLE_DEVICES=$1 python eval-client.py --domain_name reacher --task_name easy --action_repeat 4 --pre_transform_image_size 116 --work_dir ./eval --seed $2 --batch_size 512 --num_train_steps 25000 --save_tb --eval_freq 12500 
EGL_DEVICE_ID=$1 CUDA_VISIBLE_DEVICES=$1 python eval-client.py --domain_name ball_in_cup --task_name catch --action_repeat 4 --pre_transform_image_size 116 --work_dir ./eval --seed $2 --batch_size 512 --num_train_steps 25000 --save_tb --eval_freq 12500 
EGL_DEVICE_ID=$1 CUDA_VISIBLE_DEVICES=$1 python eval-client.py --domain_name finger --task_name spin --action_repeat 2 --pre_transform_image_size 116 --work_dir ./eval --seed $2 --batch_size 512 --num_train_steps 50000 --save_tb --eval_freq 12500 
EGL_DEVICE_ID=$1 CUDA_VISIBLE_DEVICES=$1 python eval-client.py --domain_name walker --task_name walk --action_repeat 2 --pre_transform_image_size 116 --work_dir ./eval --seed $2 --batch_size 512 --num_train_steps 50000 --save_tb --eval_freq 12500 
