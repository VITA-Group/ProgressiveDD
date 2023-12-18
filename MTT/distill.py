import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import epoch, get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    if args.dsa == 'True':
        args.dsa = True
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    if args.interval == 1:
        experiment_name = f'{args.dataset}_{args.model}_{args.eval_mode}_ipc{args.ipc}_max{args.max_start_epoch}_syn{args.syn_steps}_real{args.expert_epochs}_img{args.lr_img}_{args.lr_lr}_{args.lr_teacher}'

        if args.zca:
            experiment_name += '_zca'
        if len(args.tag) > 0:
            experiment_name += f'_{args.tag}'
        if args.filter_correct_samples:
            experiment_name += '_filter_correct_samples'
        elif args.filter_correct_samples_both:
            experiment_name += '_filter_correct_samples_both'
        if args.filter_easy_to_hard:
            experiment_name += f'_filter_easy_to_hard_{args.difficulty_interval}'
        if args.finetune:
            experiment_name += '_finetune'
        if args.decreasing_ipc:
            experiment_name += '_decreasing_ipc'
        if args.merge_interval:
            experiment_name += '_merge_interval'
    else:
        # use everything before the last slash of args.save_dir
        if args.override_experiment_path is None:
            experiment_name = args.save_dir.split('/')[-2]
        else:
            experiment_name = args.override_experiment_path
    
    if args.merge_interval: 
        if args.interval >= args.start_merge_interval:
            max_start_epoch = args.max_start_epoch * 2
            start_epoch_ = (args.start_merge_interval - 1) * args.max_start_epoch + 1 + (args.interval - args.start_merge_interval) * max_start_epoch
            end_epoch_ = (args.start_merge_interval - 1) * args.max_start_epoch + (args.interval - args.start_merge_interval + 1) * max_start_epoch
            # args.freq = 2
        else:
            max_start_epoch = args.max_start_epoch
            start_epoch_ = (args.interval - 1) * args.max_start_epoch + 1
            end_epoch_ = args.interval * args.max_start_epoch
    else:
        max_start_epoch = args.max_start_epoch
        start_epoch_ = (args.interval - 1) * args.max_start_epoch + 1
        end_epoch_ = args.interval * args.max_start_epoch

    if args.num_intervals > 1:
        experiment_name += f'/interval{args.interval}_epoch{start_epoch_}-{end_epoch_}'
        
    args.save_dir = os.path.join(".", args.root_log_dir, args.dataset, experiment_name)


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        n = 0
        expert_files = []
        while os.path.exists(os.path.join(args.save_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(args.save_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n != 0:
            return args
        else:
            # if best_img exists, then we have already trained this interval
            if os.path.exists(os.path.join(args.save_dir, "images_best.pt")):
                args.Iteration = 0
                print("Already trained this interval but buffer is empty. Skipping to evaluation.")

    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation",
               job_type="CleanRepo",
               config=args,
               name=experiment_name,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    if args.decreasing_ipc:
        args.ipc = [10, 10, 9, 9, 9][args.interval - 1]
    
    if args.filter_easy_to_hard:
        images = []
        labels = []
        import pickle
        forgettings = pickle.load(open(f"{args.dataset.lower()}_sorted_convnet.pkl", "rb"))
        indices = forgettings['indices']    
        forgetting = forgettings['forgetting counts']    
        sorted_forgetting = forgetting[indices]
        lower = (args.interval - 1) * args.difficulty_interval
        upper = (args.interval) * args.difficulty_interval if args.interval != args.num_intervals else 201
        for idx, (data, label) in enumerate(dst_train):

            if (sorted_forgetting[idx] >= lower) and (sorted_forgetting[idx] < upper):
                images.append(data)
                labels.append(label)
        images_all = torch.stack(images)
        labels_all = torch.from_numpy(np.array(labels, dtype=int)).reshape(-1)
        print(images_all.shape)
        indices_class = [[] for c in range(num_classes)]
        for i, lab in tqdm(enumerate(labels_all)):
            indices_class[lab].append(i)
    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]


    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)


    print(f"Syn image shape: {image_syn.shape}")
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        if args.texture:
            for c in range(num_classes):
                for i in range(args.canvas_size):
                    for j in range(args.canvas_size):
                        image_syn.data[c * args.ipc:(c + 1) * args.ipc, :, i * im_size[0]:(i + 1) * im_size[0],
                        j * im_size[1]:(j + 1) * im_size[1]] = torch.cat(
                            [get_images(c, 1).detach().data for s in range(args.ipc)])
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())
    print(args.buffer_path)
    if args.buffer_path != "./buffers":
        expert_dir = args.buffer_path
    else:
        expert_dir = os.path.join(args.buffer_path, args.dataset)
        if args.dataset == "ImageNet":
            expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
        if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
            expert_dir += "_NO_ZCA"
        expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    eval_it_pool = np.arange(args.eval_it, args.Iteration + 1, args.eval_it).tolist()
    for it in range(0, args.Iteration+1):
        save_this_it = False

        # writer.add_scalar('Progress', it, it)
        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []
                if model_eval == args.model and it == eval_it_pool[-1] and args.num_intervals > 0 and args.interval < args.num_intervals:
                    buffer = []
                    n = 0
                    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                        buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                        n += 1
                    if n == 0:
                        raise AssertionError("No buffers detected at {}".format(expert_dir))
                    iterations = len(buffer)
                    trajectories = []
                    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

                    image_syn = torch.load(os.path.join(args.save_dir, 'images_best.pt'))
                    label_syn = torch.load(os.path.join(args.save_dir, 'labels_best.pt'))
                    print('load best images and labels from %s'%args.save_dir)

                    try:
                        args.lr_net = best_lr
                        print('best lr: ', args.lr_net)
                    except:
                        args.lr_net = torch.tensor(args.lr_teacher).to(args.device)

                    if args.finetune:
                        # finetune the synthetic images by further matching the end of the expert trajectory
                        print('Finetune the synthetic images by further matching the end of the expert trajectory')
                        image_syn = image_syn.detach().to(args.device).requires_grad_(True)
                        syn_lr = torch.tensor(args.lr_net).to(args.device).requires_grad_(True)
                        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
                        optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
                        optimizer_img.zero_grad()
                        
                        # for args.finetune_steps steps
                        for ft_it in range(args.finetune_steps):
                            # randomly select an expert trajectory
                            i = np.random.randint(0, len(buffer))
                            # get the end of the expert trajectory
                            starting_params = buffer[i][0]

                            # get a random model
                            student_net = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                            # load the start of the expert trajectory
                            for p, q in zip(student_net.state_dict().items(), starting_params):
                                p[1].copy_(q[1])


                            student_net = ReparamModule(student_net)

                            if args.distributed:
                                student_net = torch.nn.DataParallel(student_net)

                            student_net.train()
                            
                            # train the model on the synthetic images with evaluate_synset
                            num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

                            # get the end of the trained trajectory
                            student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
                            starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

                            syn_images = image_syn

                            y_hat = label_syn.to(args.device)

                            param_loss_list = []
                            param_dist_list = []
                            indices_chunks = []

                            for step in range(1, args.syn_steps*2+1):
                                if not indices_chunks:
                                    indices = torch.randperm(len(syn_images))
                                    indices_chunks = list(torch.split(indices, args.batch_syn))

                                these_indices = indices_chunks.pop()

                                x = syn_images[these_indices]
                                this_y = y_hat[these_indices]

                                if args.texture:
                                    x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                                    this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

                                if args.dsa and (not args.no_aug):
                                    x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

                                if args.distributed:
                                    forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                                else:
                                    forward_params = student_params[-1]
                                x = student_net(x, flat_param=forward_params)
                                ce_loss = criterion(x, this_y)

                                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

                                student_params.append(student_params[-1] - syn_lr * grad)

                            param_loss_list = []

                            target_params = buffer[i][max_start_epoch*2]
                            target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

                            param_loss = torch.tensor(0.0).to(args.device)
                            param_dist = torch.tensor(0.0).to(args.device)

                            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
                            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

                            param_loss /= num_params
                            param_dist /= num_params

                            param_loss /= param_dist

                            grand_loss = param_loss

                            optimizer_img.zero_grad()
                            optimizer_lr.zero_grad()

                            grand_loss.backward()

                            optimizer_img.step()
                            optimizer_lr.step()

                            wandb.log({"Finetune_Loss": grand_loss.detach().cpu(),
                                       "Finetune_Learning_Rate": syn_lr.detach().cpu()})
                            
                            for _ in student_params:
                                del _

                            if ft_it%10 == 0:
                                print('%s iter = %04d, loss = %.4f' % (get_time(), ft_it, grand_loss.item()))
                else:
                    iterations = args.num_eval
                    args.lr_net = syn_lr.item()

                if it == eval_it_pool[-1] and (args.filter_correct_samples or args.filter_correct_samples_both):
                    correct = torch.zeros(len(dst_train)).cuda()
                    images = []
                    labels = []
                    for it_eval in range(iterations):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                        for b, p in zip(buffer[it_eval][0], net_eval.state_dict().items()):
                            # load the same weights
                            p[1].copy_(b.data)

                        eval_labs = label_syn
                        with torch.no_grad():
                            image_save = image_syn
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                        teacher_net, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)

                        loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, num_workers=0)
                        cur = 0

                        for data, label in loader:
                            data = data.cuda()
                            label = label.cuda()
                            output = teacher_net(data)
                            output = torch.argmax(output, 1)
                            correct_batch = output == label
                            correct[cur:cur + output.shape[0]] += correct_batch
                            cur += output.shape[0]
                            
                    correct = correct.cpu().numpy()
                        
                    for idx, (data, label) in enumerate(dst_train):
                        if (correct[idx] < iterations) and (0 < correct[idx]):
                            images.append(data)
                            labels.append(label)
                        elif (not args.filter_correct_samples_both) and (0 == correct[idx]):
                            images.append(data)
                            labels.append(label)
                        else:
                            if torch.rand(1) < 0.2:
                                images.append(data)
                                labels.append(label)
                    images = torch.stack(images)
                    labels = torch.from_numpy(np.array(labels, dtype=int)).reshape(-1)

                    dst_train = torch.utils.data.TensorDataset(images, labels)
                    dst_train.targets = labels
                    dst_train.data = images
                    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

                for it_eval in range(iterations):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                    for b, p in zip(buffer[it_eval][0], net_eval.state_dict().items()):
                        # load the same weights
                        p[1].copy_(b.data)

                    eval_labs = label_syn
                    with torch.no_grad():
                        image_save = image_syn
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                    teacher_net, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)

                    if model_eval == args.model and it == eval_it_pool[-1] and args.num_intervals > 0 and args.interval < args.num_intervals:
                        teacher_net.train()
                        lr = args.lr_teacher
                        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)  # optimizer_img for synthetic data
                        teacher_optim.zero_grad()

                        timestamps = []

                        if args.merge_interval: 
                            if args.interval >= 3:
                                max_start_epoch = args.max_start_epoch * 2
                            else:
                                max_start_epoch = args.max_start_epoch
                        else:
                            max_start_epoch = args.max_start_epoch
                        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])
                        for e in range(max_start_epoch+args.expert_epochs+1):
                            
                            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                                        criterion=criterion, args=args, aug=True)

                            test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                                        criterion=criterion, args=args, aug=False)

                            print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}".format(it_eval, e, train_acc, test_acc))

                            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

                        trajectories.append(timestamps)

                        if len(trajectories) == args.save_interval:
                            n = 0
                            while os.path.exists(os.path.join(args.save_dir, "replay_buffer_{}.pt".format(n))):
                                n += 1
                            print("Saving {}".format(os.path.join(args.save_dir, "replay_buffer_{}.pt".format(n))))
                            torch.save(trajectories, os.path.join(args.save_dir, "replay_buffer_{}.pt".format(n)))
                            trajectories = []

                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    best_lr = syn_lr.clone().detach()
                    save_this_it = True
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

                if it == args.Iteration:
                    wandb.finish()
                    return args

        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image_syn.cuda()

                torch.save(image_save.cpu(), os.path.join(args.save_dir, "images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(args.save_dir, "labels_{}.pt".format(it)))
                torch.save(syn_lr.clone().detach().cpu(), os.path.join(args.save_dir, "syn_lr_{}.pt".format(it)))

                if save_this_it:
                    print('save best images and labels to %s at iteration %d'%(args.save_dir, it))
                    torch.save(image_save.cpu(), os.path.join(args.save_dir, "images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(args.save_dir, "labels_best.pt".format(it)))
                    torch.save(syn_lr.clone().detach().cpu(), os.path.join(args.save_dir, "syn_lr_best.pt".format(it)))

                    print('lr: ', args.lr_net)

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(args.save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)

        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        if args.freq > 1:
            start_epoch = random.choice(np.arange(start_sampling_epoch, max_start_epoch + start_sampling_epoch, args.freq))
        else:
            start_epoch = np.random.randint(start_sampling_epoch, max_start_epoch + start_sampling_epoch)
        
        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn

        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()


            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            if args.texture:
                x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]

            if args.distributed:
                forward_params = forward_params.cpu()
                x = x.cpu()
            
            x = student_net(x, flat_param=forward_params)
            # print(x)
            # print(x.shape)
            ce_loss = criterion(x, this_y)
            
            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
            student_params.append(student_params[-1] - syn_lr * grad)
        
        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)


        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        optimizer_img.step()
        optimizer_lr.step()

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

    wandb.finish()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--save_dir', type=str, default=None)

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    # add number of intervals
    parser.add_argument('--num_intervals', type=int, default=1, help='number of intervals to use')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)

    parser.add_argument('--filter-correct-samples', action='store_true')
    parser.add_argument('--filter-correct-samples-both', action='store_true')
    # finetune 
    parser.add_argument('--finetune', action='store_true', help='finetune the synthetic images by further matching the end of the expert trajectory')
    parser.add_argument('--finetune_steps', type=int, default=1000, help='number of finetuning steps')
    parser.add_argument('--decreasing-ipc', action="store_true")
    parser.add_argument('--merge-interval', action="store_true")
    parser.add_argument('--start_interval', type=int, default=1)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--override_experiment_path', type=str, default=None)

    parser.add_argument('--filter_easy_to_hard', action='store_true')
    parser.add_argument('--difficulty-interval', type=int, default=10)
    parser.add_argument('--freq', type=int, default=1)
    parser.add_argument('--root_log_dir', type=str, default='logged_files')
    parser.add_argument('--start_merge_interval', type=int, default=3)
    parser.add_argument('--start_sampling_epoch', type=int, default=0)

    args = parser.parse_args()
    
    iterations = args.Iteration
    start_sampling_epoch = args.start_sampling_epoch
    for interval in range(args.start_interval, args.num_intervals+1):
        args.interval = interval

        if args.dataset == 'CIFAR10' and args.ipc == 1 and interval > 1:
            args.lr_lr = 1e-05

        args.Iteration = iterations

        args = main(args)
        args.buffer_path = args.save_dir


