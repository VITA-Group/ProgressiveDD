# MTT-based experiments

## Commands

### Generating training trajectories

python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=100 --zca --data_path=data

### CIFAR-10 IPC=50

python distill_increase.py --dataset=CIFAR10 --model=ConvNet --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=15 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --num_intervals 5