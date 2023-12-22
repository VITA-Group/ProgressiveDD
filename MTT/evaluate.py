

import argparse
import copy
import numpy as np
import os
import torch
import torch.nn as nn
import wandb
from utils import epoch, evaluate_synset, get_dataset, get_eval_pool, get_network, ParamDiffAug
import torch.nn.utils.prune as prune
parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
parser.add_argument('--model', type=str, default='ConvNet', help='model')
parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
parser.add_argument('--eval_mode', type=str, default='S',
                    help='eval_mode, check utils.py for more info')
parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')

# training parameters
parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                    help='differentiable Siamese augmentation strategy')

parser.add_argument('--data_path', type=str, default='data', help='dataset path')
parser.add_argument('--save_path', type=str, default='logged_files', help='save path')

parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
parser.add_argument('--texture', action='store_true', help="will distill textures instead")
parser.add_argument('--prune', action='store_true')
parser.add_argument('--sparse_to_dense', action='store_true')

# evaluation parameters
parser.add_argument('--num_intervals', type=int, default=2, help='how many intervals to evaluate')

# buffer parameters
parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
parser.add_argument('--mom', type=float, default=0, help='momentum')
parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
parser.add_argument('--save_interval', type=int, default=10)
parser.add_argument('--override_interval_names', type=str, default=None)
parser.add_argument('--override_load_path', type=str, default=None)
parser.add_argument('--buffer_path', type=str, default='buffers')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--save_model_prefix', type=str, default='')


args = parser.parse_args()

index = [1,0,2,3,5,4,6,7,9,8,10,11,12,13]

print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.dsa = True

experiment_name = f'{args.dataset}_{args.model}_{args.eval_mode}_ipc{args.ipc}_it{args.num_intervals}_max{args.max_start_epoch}_syn{args.syn_steps}_real{args.expert_epochs}_img{args.lr_img}_{args.lr_lr}_{args.lr_teacher}'
if args.zca:
    experiment_name += '_zca'
if args.override_load_path is not None:
    experiment_name = args.override_load_path

wandb.init(sync_tensorboard=False,
            project="DatasetDistillation",
            job_type="CleanRepo",
            config=args,
            name=experiment_name,
            )

args = type('', (), {})()

# initialize dsa parameters
args.dsa_param = ParamDiffAug()

for key in wandb.config._items:
    setattr(args, key, wandb.config._items[key])

# load the dataset
channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

# initialize the evaluation models
net_eval_pool = {}
expert_dir = os.path.join(args.buffer_path, args.dataset)
if args.dataset == "ImageNet":
    expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
    expert_dir += "_NO_ZCA"
expert_dir = os.path.join(expert_dir, args.model)
print("Try to load from {}".format(expert_dir))
expert_files = []
n = 0
while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
    expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
    n += 1
print(expert_files)
import random
file_idx = 0
expert_idx = 0
# random.shuffle(expert_files)

try:
    print("loading file {}".format(expert_files[file_idx]))
    buffer = torch.load(expert_files[file_idx])
    # random.shuffle(buffer)
except:
    buffer = []

for model_eval in model_eval_pool:
    net_eval_pool[model_eval] = []

    if model_eval == args.model:
        num_eval = args.num_experts
    else:
        num_eval = args.num_eval

    for it_eval in range(num_eval):
        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
        if args.sparse_to_dense or args.prune:
            buffer_ = [buffer[it_eval][0][idx] for idx in index]
        else:
            try:
                buffer_ = buffer[it_eval][0]
            except:
                pass
        if len(buffer) > 0:
            for b, p in zip(buffer_, net_eval.state_dict().items()):
                # load the same weights
                p[1].copy_(b.data)
        if args.prune:
            state_dict = net_eval.state_dict()
            for name, m in net_eval.named_modules():
                if isinstance(m, nn.Conv2d):
                    prune.custom_from_mask(m, 'weight', state_dict[name + ".weight"].abs() > 1e-9)
                    print(torch.mean((state_dict[name + ".weight"].abs() < 1e-9).float()))
        net_eval_pool[model_eval].append(net_eval)

best_acc = {m: 0 for m in model_eval_pool}
best_std = {m: 0 for m in model_eval_pool}

syn_lr = torch.tensor(args.lr_teacher).to(args.device)

''' Evaluate synthetic data '''
for model_eval in model_eval_pool:
    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s'%(args.model, model_eval))
    print('DSA augmentation strategy: \n', args.dsa_strategy)
    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

    if model_eval == args.model:
        num_eval = args.num_experts
    else:
        num_eval = args.num_eval

    it = 1
    experiment_name = f'{args.dataset}_{args.model}_{args.eval_mode}_ipc{args.ipc}_max{args.max_start_epoch}_syn{args.syn_steps}_real{args.expert_epochs}_img{args.lr_img}_{args.lr_lr}_{args.lr_teacher}'
    if args.zca:
        experiment_name += '_zca'
    if args.override_load_path is not None:
        experiment_name = args.override_load_path
    max_start_epoch = args.max_start_epoch * it

    for it_eval in range(num_eval):
        accs_test = []
        accs_train = []
        init_state = None
        for it in range(1, args.num_intervals + 1):
            # load the synthetic data
            experiment_name = f'{args.dataset}_{args.model}_{args.eval_mode}_ipc{args.ipc}_max{args.max_start_epoch}_syn{args.syn_steps}_real{args.expert_epochs}_img{args.lr_img}_{args.lr_lr}_{args.lr_teacher}'
            if args.zca:
                experiment_name += '_zca'
            if args.override_load_path is not None:
                experiment_name = args.override_load_path
            max_start_epoch = args.max_start_epoch * it
            interval_name = f'interval{it}_epoch{args.max_start_epoch * (it - 1) + 1}-{max_start_epoch}'
            if args.override_interval_names is not None:
                interval_name = args.override_interval_names.split(".")[it - 1]
            print(f"Load from {os.path.join(args.save_path, args.dataset, experiment_name, interval_name, 'images_best.pt')}")
            image_syn = torch.load(os.path.join(args.save_path, args.dataset, experiment_name, interval_name, 'images_best.pt'))
            label_syn = torch.load(os.path.join(args.save_path, args.dataset, experiment_name, interval_name, 'labels_best.pt'))
            syn_lr = torch.load(os.path.join(args.save_path, args.dataset, experiment_name, interval_name, 'syn_lr_best.pt'))
            # syn_lr = torch.tensor(0.01)
            net_eval = net_eval_pool[model_eval][it_eval]
            eval_labs = label_syn
            with torch.no_grad():
                image_save = image_syn
            image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification
            args.lr_net = syn_lr.item()
            
            net_eval, acc_train, acc_test, optimizer = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture,
            save_model=args.save_model, save_model_prefix=args.save_model_prefix + f"_{it}", stage=it, warmup=0,
            return_optimizer=True, init_state=init_state)
            init_state = optimizer.state
            net_eval_pool[model_eval][it_eval] = net_eval
        accs_test.append(acc_test)
        accs_train.append(acc_train)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)
        if acc_test_mean > best_acc[model_eval]:
            best_acc[model_eval] = acc_test_mean
            best_std[model_eval] = acc_test_std
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
        wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
        wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
        wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
        wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)
