CUDA_VISIBLE_DEVICES=7 python distill.py --dataset=CIFAR10 --model=ConvNet --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=5 --zca --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --zca

CUDA_VISIBLE_DEVICES=7 python evaluate.py --dataset=CIFAR10 --model=ConvNet --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=5 --zca --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --zca --num_interval=2

CUDA_VISIBLE_DEVICES=6 python distill.py --dataset=CIFAR10 --model=ConvNet --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=15 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --zca