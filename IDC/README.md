# IDC-based experiments

## Commands

### CIFAR-10 IPC=10

python -u condense_iterative.py --reproduce  -d cifar100 -f 2 --ipc 2 --data_dir data --tag iterative_increase --niter 1000 --start-interval 0

### CIFAR-10 IPC=50

python -u condense_iterative.py --reproduce  -d cifar100 -f 2 --ipc 10 --data_dir data --tag iterative_increase --niter 1000 --start-interval 0
