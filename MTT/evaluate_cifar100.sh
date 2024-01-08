CUDA_VISIBLE_DEVICES=1 python evaluate.py --dataset=CIFAR100 --model=ConvNet --ipc=2 --syn_steps=20 --expert_epochs=3 --max_start_epoch=20 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --num_intervals 5 --num_experts 5 --override_load_path CIFAR100_ConvNet_S_ipc2_max20_syn20_real3_img1000.0_1e-05_0.01_increase_zca --save_path logged_files --epoch_eval_train 500 --save_model_prefix temp


CUDA_VISIBLE_DEVICES=1 python evaluate.py --dataset=CIFAR100 --model=ConvNet --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=20 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --num_intervals 5 --num_experts 5 --override_load_path CIFAR100_ConvNet_S_ipc10_max20_syn20_real2_img1000.0_1e-05_0.01_increase_zca --save_path logged_files --epoch_eval_train 500 --save_model_prefix temp

