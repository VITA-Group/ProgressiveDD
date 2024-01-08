# MTT-based experiments

## Pretrained buffers and pre-synthesized datasets

The buffers can be found in [this link](https://drive.google.com/drive/folders/1ZKamN6ledxEnmLyya4Mbl57Fu_H1Axb6?usp=sharing).

The synthesized samples can be found in [this link](https://drive.google.com/drive/folders/1OKnCiKO78-MOuMXRJWGyE3uV69t2kIMV?usp=sharing). 

## Commands

### Generating training trajectories

python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=100 --zca --data_path=data

### Dataset Synthesis

Please refer to `train_cifar10.sh` and `train_cifar100.sh`.

### Dataset Evaluation

Please refer to `evaluate_cifar10.sh` and `evaluate_cifar100.sh`