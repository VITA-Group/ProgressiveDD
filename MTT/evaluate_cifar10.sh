
CUDA_VISIBLE_DEVICES=1 python evaluate.py --dataset=CIFAR10 --model=ConvNet --ipc=2 --syn_steps=50 --expert_epochs=2 --max_start_epoch=5 --zca --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --num_intervals 5 --num_experts 5 --override_load_path CIFAR10_ConvNet_S_ipc2_max5_syn50_real2_img1000.0_1e-07_0.01_increase_zca --save_path logged_files --epoch_eval_train 500 --save_model_prefix temp

CUDA_VISIBLE_DEVICES=1 python evaluate.py --dataset=CIFAR10 --model=ConvNet --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=15 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --num_intervals 5 --num_experts 5 --override_load_path CIFAR10_ConvNet_S_ipc10_max15_syn30_real2_img1000.0_1e-05_0.01_increase_zca --save_path logged_files --epoch_eval_train 500 --save_model_prefix temp




