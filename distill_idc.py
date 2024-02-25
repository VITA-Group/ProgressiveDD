import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion, transform_tiny_imagenet
from data import TensorDataset, ImageFolder, save_img
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from train import define_model, train_epoch, train_only
from test import test_data, load_ckpt, test_data_with_previous
from misc.augment import DiffAug
from misc import utils
from math import ceil
import glob
import pickle


class Synthesizer():
    """Condensed data class
    """
    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor([np.ones(self.ipc, dtype=int) * i for i in range(int(nclass))],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.targets[i]].append(i)

        print("\nDefine synthetic data: ", self.data.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            print("Mixed initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc * self.factor**2)
                img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                                       w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor**2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor**2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):
        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)
        return data, target

    def loader(self, args, augment=True):
        """Data loader for condensed data
        """
        if args.dataset == 'imagenet':
            train_transform, _ = transform_imagenet(augment=augment,
                                                    from_tensor=True,
                                                    size=0,
                                                    rrc=args.rrc,
                                                    rrc_size=self.size[0])
        elif args.dataset == 'tiny-imagenet':
            train_transform, _ = transform_tiny_imagenet(
                augment=augment,
                from_tensor=True,
                size=0,
                rrc=args.rrc,
                rrc_size=self.size[0])
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(augment=augment, from_tensor=True)
        elif args.dataset == 'svhn':
            train_transform, _ = transform_svhn(augment=augment, from_tensor=True)
        elif args.dataset == 'mnist':
            train_transform, _ = transform_mnist(augment=augment, from_tensor=True)
        elif args.dataset == 'fashion':
            train_transform, _ = transform_fashion(augment=augment, from_tensor=True)

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            idx_to = self.ipc * (c + 1)
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            data, target = self.decode(data, target)

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        train_dataset = TensorDataset(data_dec.cpu(), target_dec.cpu(), train_transform)

        print("Decode condensed data: ", data_dec.shape)
        nw = 0 if not augment else args.workers
        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             persistent_workers=nw > 0)
        return train_loader

    def test(self, args, val_loader, logger, bench=True):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment)
        test_data(args, loader, val_loader, test_resnet=False, logger=logger)

        if bench and not (args.dataset in ['mnist', 'fashion']):
            test_data(args, loader, val_loader, test_resnet=True, logger=logger)

    def test_with_previous(self, args, val_loader, previous_train_loaders, logger, bench=True):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment)
        test_data_with_previous(args, loader, val_loader, previous_train_loaders, test_resnet=False, logger=logger)

        if bench and not (args.dataset in ['mnist', 'fashion']):
            test_data_with_previous(args, loader, val_loader, previous_train_loaders, test_resnet=True, logger=logger)


def load_resized_data(args):
    """Load original training data (fixed spatial size and without augmentation) for condensation
    """
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transforms.ToTensor())
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 100

    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                      split='train',
                                      transform=transforms.ToTensor())
        train_dataset.targets = train_dataset.labels

        normalize = transforms.Normalize(mean=MEANS['svhn'], std=STDS['svhn'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                    split='test',
                                    transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(args.data_dir, train=True, transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['mnist'], std=STDS['mnist'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir,
                                              train=True,
                                              transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['fashion'], std=STDS['fashion'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.FashionMNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10
    elif args.dataset == 'tiny-imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')
        print(traindir)
        # We preprocess images to the fixed size (default: 224)
        resize = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.PILToTensor()
        ])

        if args.load_memory:  # uint8
            transform = None
            load_transform = resize
        else:
            transform = transforms.Compose([resize, transforms.ConvertImageDtype(torch.float)])
            load_transform = None

        _, test_transform = transform_tiny_imagenet(size=64)
        train_dataset = ImageFolder(traindir,
                                    transform=transform,
                                    nclass=args.nclass,
                                    phase=args.phase,
                                    seed=args.dseed,
                                    load_memory=args.load_memory,
                                    load_transform=load_transform)
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  phase=args.phase,
                                  seed=args.dseed,
                                  load_memory=False)
    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        # We preprocess images to the fixed size (default: 224)
        resize = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.PILToTensor()
        ])

        if args.load_memory:  # uint8
            transform = None
            load_transform = resize
        else:
            transform = transforms.Compose([resize, transforms.ConvertImageDtype(torch.float)])
            load_transform = None

        _, test_transform = transform_imagenet(size=args.size)
        train_dataset = ImageFolder(traindir,
                                    transform=transform,
                                    nclass=args.nclass,
                                    phase=args.phase,
                                    seed=args.dseed,
                                    load_memory=args.load_memory,
                                    load_transform=load_transform)
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  phase=args.phase,
                                  seed=args.dseed,
                                  load_memory=False)

    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    assert train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]  # width check

    return train_dataset, val_loader


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    normalize = utils.Normalize(mean=MEANS[args.dataset], std=STDS[args.dataset], device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_


def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss


def matchloss(args, img_real, img_syn, lab_real, lab_syn, model):
    """Matching losses (feature or gradient)
    """
    loss = None

    if args.match == 'feat':
        with torch.no_grad():
            feat_tg = model.get_feature(img_real, args.idx_from, args.idx_to)
        feat = model.get_feature(img_syn, args.idx_from, args.idx_to)

        for i in range(len(feat)):
            loss = add_loss(loss, dist(feat_tg[i].mean(0), feat[i].mean(0), method=args.metric))

    elif args.match == 'grad':
        criterion = nn.CrossEntropyLoss()

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))

        output_syn = model(img_syn)
        loss_syn = criterion(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))

    return loss


def pretrain_sample(args, model, verbose=False):
    """Load pretrained networks
    """
    folder_base = f'./pretrained/{args.datatag}/{args.modeltag}_cut'
    folder_list = glob.glob(f'{folder_base}*')
    tag = np.random.randint(len(folder_list))
    folder = folder_list[tag]

    epoch = args.pt_from
    if args.pt_num > 1:
        epoch = np.random.randint(args.pt_from, args.pt_from + args.pt_num)
    ckpt = f'checkpoint{epoch}.pth.tar'

    file_dir = os.path.join(folder, ckpt)
    load_ckpt(model, file_dir, verbose=verbose)


def condense(args, logger, device='cuda'):
    """Optimize condensed data
    """
    # Define real dataset and loader
    print(args)
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    if args.start_interval > 0:
        previous_images, previous_labels = torch.load(os.path.join(args.save_dir, f'interval_{args.start_interval - 1}_data.pt'))
    else:
        previous_images = None
        previous_labels = None
    for interval_idx in range(args.start_interval, args.num_intervals):
        # Define syn dataset
        print("=" * 20)
        print(f"Begin interval: {interval_idx}")
        print("=" * 20)
        images = []
        labels = []
        if args.filter_easy_to_hard:
            trainset, val_loader = load_resized_data(args)

            forgettings = pickle.load(open("cifar10_sorted_convnet.pkl", "rb"))
            indices = forgettings['indices']    
            forgetting = forgettings['forgetting counts']    
            sorted_forgetting = forgetting[indices]
            lower = interval_idx * args.difficulty_interval
            upper = (interval_idx + 1) * args.difficulty_interval if interval_idx != args.num_intervals - 1 else 201
            for idx, (data, label) in enumerate(trainset):
                
                if (sorted_forgetting[idx] >= lower) and (sorted_forgetting[idx] < upper):
                    images.append(data)
                    labels.append(label)
            images = torch.stack(images)
            labels = torch.from_numpy(np.array(labels, dtype=int)).reshape(-1)
            print(images.shape)

            trainset = torch.utils.data.TensorDataset(images, labels)
            trainset.targets = labels
            trainset.data = images
            trainset.nclass = args.nclass
        
        torch.manual_seed(interval_idx)
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
        synset = Synthesizer(args, nclass, nch, hs, ws)
        synset.init(loader_real, init_type=args.init)
        
        if previous_images is not None:
            previous_images = previous_images.to(synset.data.data.device)
            previous_labels = previous_labels.to(synset.targets.data.device)
            with torch.no_grad():
                previous_images = previous_images.reshape(synset.data.shape[0], -1, *synset.data.shape[1:])
                new_data = torch.cat([previous_images, synset.data.unsqueeze(1)], 1)
                new_targets = torch.cat([previous_labels.reshape(synset.targets.shape[0], -1), synset.targets.unsqueeze(1)], 1)
                grad_mask = torch.cat([torch.zeros_like(previous_images), torch.ones_like(synset.data).unsqueeze(1)], 1).reshape(-1, *new_data.shape[2:])
                new_data = new_data.reshape(-1, *new_data.shape[2:])
                
                new_targets = new_targets.reshape(-1)
                synset.data = torch.tensor(new_data,
                                dtype=torch.float,
                                requires_grad=True,
                                device=synset.device)
                synset.targets = torch.tensor(new_targets,
                                dtype=torch.long,
                                requires_grad=False,
                                device=synset.device)
                synset.ipc = synset.data.shape[0] // nclass
                print(synset.ipc)
        else:
            grad_mask = None
        save_img(os.path.join(args.save_dir, f'interval_{interval_idx}_init.png'),
                synset.data,
                unnormalize=False,
                dataname=args.dataset)

        # Define augmentation function
        aug, aug_rand = diffaug(args)
        save_img(os.path.join(args.save_dir, f'interval_{interval_idx}_aug.png'),
                aug(synset.sample(0, max_size=args.batch_syn_max)[0]),
                unnormalize=True,
                dataname=args.dataset)

        
        prev_loaders = []
        if interval_idx >= 1:
            for i in range(interval_idx):
                prev_data, prev_targets = torch.load(os.path.join(args.save_dir, f'interval_{i}_data.pt'))
                old_ipc = int(args.ipc)
                new_ipc = old_ipc * (i + 1)
                args.ipc = new_ipc
                synset_old = Synthesizer(args, nclass, nch, hs, ws)
                synset_old.init(loader_real, init_type=args.init)
                with torch.no_grad():
                    synset_old.data.copy_(prev_data)
                    synset_old.targets.copy_(prev_targets)
                prev_loader = synset_old.loader(args, args.augment)
                prev_loaders.append(prev_loader)
                args.ipc = old_ipc
                print(args.ipc)
            if args.filter_correct_samples or args.filter_correct_samples_both:
                correct = torch.zeros(len(trainset)).cuda()
                for idx in range(10):
                    model = define_model(args, nclass).to(device)
                    model.train()
                    for j in range(interval_idx):
                        prev_loader = prev_loaders[j]
                        best_acc1, acc1, return_weights = train_only(args, model, prev_loader, True, logger=logger)
                        model.load_state_dict(return_weights)
                    loader_acc_evaluate = ClassDataLoader(trainset,
                                    batch_size=args.batch_real,
                                    num_workers=args.workers,
                                    pin_memory=True)
                    cur = 0
                    for data, label in loader_acc_evaluate:
                        data = data.cuda()
                        label = label.cuda()
                        output = model(data)
                        output = torch.argmax(output, 1)
                        correct_batch = output == label
                        correct[cur:cur + output.shape[0]] += correct_batch
                        cur += output.shape[0]
                
                correct = correct.cpu().numpy()
                
                for idx, (data, label) in enumerate(trainset):
                    if (correct[idx] < 10) and (0 < correct[idx]):
                        images.append(data)
                        labels.append(label)
                    elif (not args.filter_correct_samples_both) and (0 == correct[idx]):
                        images.append(data)
                        labels.append(label)
                images = torch.stack(images)
                labels = torch.from_numpy(np.array(labels, dtype=int)).reshape(-1)
                print(images.shape)

                trainset = torch.utils.data.TensorDataset(images, labels)
                trainset.targets = labels
                trainset.data = images
                trainset.nclass = args.nclass

        if not args.test:
            synset.test_with_previous(args, val_loader, prev_loaders, logger, bench=False)

        # Data distillation
        print(synset.parameters())

        optim_img = torch.optim.SGD(synset.parameters(), lr=args.lr_img, momentum=args.mom_img)

        ts = utils.TimeStamp(args.time)
        n_iter = args.niter * 100 // args.inner_loop
        it_log = max(n_iter // 50, 1)
        it_test = np.arange(0, n_iter, 50)
        # [n_iter // 10, n_iter // 5, n_iter // 4, n_iter // 3, n_iter // 2, n_iter // 3 * 2, n_iter // 4 * 3, n_iter]

        logger(f"\nStart condensing with {args.match} matching for {n_iter} iteration")
        args.fix_iter = max(1, args.fix_iter)
        for it in range(n_iter):
            if it % args.fix_iter == 0:
                model = define_model(args, nclass).to(device)
                model.train()
                if interval_idx > 0:
                    for i in range(interval_idx):
                        prev_loader = prev_loaders[i]
                        best_acc1, acc1, return_weights = train_only(args, model, prev_loader, True, logger=logger, epochs=args.epochs / 5)
                        model.load_state_dict(return_weights)
                
                optim_net = optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
                criterion = nn.CrossEntropyLoss()

                if args.pt_from >= 0:
                    pretrain_sample(args, model)
                if args.early > 0:
                    for _ in range(args.early):
                        train_epoch(args,
                                    loader_real,
                                    model,
                                    criterion,
                                    optim_net,
                                    aug=aug_rand,
                                    mixup=args.mixup_net)

            loss_total = 0
            synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
            for ot in range(args.inner_loop):
                ts.set()

                # Update synset
                for c in range(nclass):
                    img, lab = loader_real.class_sample(c)
                    img_syn, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
                    ts.stamp("data")

                    n = img.shape[0]
                    img_aug = aug(torch.cat([img, img_syn]))
                    ts.stamp("aug")

                    loss = matchloss(args, img_aug[:n], img_aug[n:], lab, lab_syn, model)
                    loss_total += loss.item()
                    ts.stamp("loss")

                    optim_img.zero_grad()
                    loss.backward()
                    if grad_mask is not None:
                        synset.data.grad.data.mul_(grad_mask)

                    optim_img.step()
                    ts.stamp("backward")

                # Net update
                if args.n_data > 0:
                    for _ in range(args.net_epoch):
                        train_epoch(args,
                                    loader_real,
                                    model,
                                    criterion,
                                    optim_net,
                                    n_data=args.n_data,
                                    aug=aug_rand,
                                    mixup=args.mixup_net)
                ts.stamp("net update")

                if (ot + 1) % 10 == 0:
                    ts.flush()

            # Logging
            if it % it_log == 0:
                logger(
                    f"{utils.get_time()} (Iter {it:3d}) loss: {loss_total/nclass/args.inner_loop:.1f}")
            if (it + 1) in it_test:
                previous_images = synset.data.data.clone()
                previous_labels = synset.targets.data.clone()
                save_img(os.path.join(args.save_dir, f'interval_{interval_idx}_img{it+1}.png'),
                        synset.data,
                        unnormalize=False,
                        dataname=args.dataset)

                # It is okay to clamp data to [0, 1] at here.
                # synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
                if args.override_save_dir is not None:
                    os.makedirs(args.override_save_dir, exist_ok=True)
                    torch.save(
                        [synset.data.detach().cpu(), synset.targets.cpu()],
                        os.path.join(args.override_save_dir, f'interval_{interval_idx}_data.pt'))
                else:
                    torch.save(
                        [synset.data.detach().cpu(), synset.targets.cpu()],
                        os.path.join(args.save_dir, f'interval_{interval_idx}_data.pt'))
                print("img and data saved!")

                if not args.test:
                    synset.test_with_previous(args, val_loader, prev_loaders, logger, bench=False)


if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn
    import json

    assert args.ipc > 0

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.save_dir)
    if args.override_save_dir is None:
        logger = Logger(args.save_dir)
        logger(f"Save dir: {args.save_dir}")
        with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        os.makedirs(args.override_save_dir, exist_ok=True)
        logger = Logger(args.override_save_dir)
        logger(f"Save dir: {args.override_save_dir}")
        with open(os.path.join(args.override_save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    

    condense(args, logger)