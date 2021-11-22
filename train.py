#!/usr/bin/env python
"""
train ALT
Created by anonymous on 2021-11-21
"""

import os
import sys
sys.path.append(os.path.abspath(''))
import random
import argparse
import torch 
import torchvision 
from torchvision.datasets import ImageFolder
from torch.utils.data import dataloader
from lib.datasets import get_dataset, pacs

from lib.datasets.transforms import GreyToColor, IdentityTransform, ToGrayScale, LaplacianOfGaussianFiltering
import torchvision.transforms as transforms

from trainer_alt import *
from lib.networks import get_network
from metann import Learner

def main(args):
    # GPU and random seed
    print("Random Seed: ", args.rand_seed)
    if args.rand_seed is not None:
        random.seed(args.rand_seed)
        torch.manual_seed(args.rand_seed)
        print(args.gpu_ids, type(args.gpu_ids))
        if type(args.gpu_ids) is list and len(args.gpu_ids) >= 0:
            torch.cuda.manual_seed_all(args.rand_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.rand_seed)
        torch.set_num_threads(1)

    # DATALOADERS
    if args.data_name=='pacs':
        assert args.n_classes==7
        data_dir = './data/PACS/'
        domains = ['photo', 'art_painting', 'cartoon', 'sketch']
        trg_domains = [dd for dd in domains if dd!=args.source]
        stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif args.data_name=='digits':
        assert args.n_classes==10
        data_dir = "./data/"
        domains = ['mnist10k', 'mnist_m', 'svhn', 'usps', 'synth']
        trg_domains = [dd for dd in domains if dd!=args.source]
        stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif args.data_name=='officehome':
        assert args.n_classes==65
        data_dir = './data/OfficeHome/'
        domains = ['real', 'art', 'clipart', 'product']
        trg_domains = [dd for dd in domains if dd!=args.source]
        stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    print(args.data_name)
    print("SRC:{}; TRG:{}".format(args.source, domains))

    # transforms
    trans_list = []
    trans_list.append(
        transforms.RandomResizedCrop(args.image_size, scale=(0.5, 1))
        )
    if args.colorjitter:
        trans_list.append(transforms.ColorJitter(*[args.colorjitter] * 4))
    if args.data_name != 'digits':
        trans_list.append(transforms.RandomHorizontalFlip())
    trans_list.append(transforms.ToTensor())
    if args.data_name=='digits':
        trans_list.append(GreyToColor())
    trans_list.append(transforms.Normalize(*stats))
    train_transform = transforms.Compose(trans_list)

    test_transform  = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        GreyToColor() if args.data_name=='digits' else IdentityTransform(),
        transforms.Normalize(*stats)
        ])

    ## datasets
    print("\n=========Preparing Data=========")
    assert args.source in domains, 'allowed data_name {}'.format(domains)

    if args.data_name=='pacs':
        trainset = pacs.PACS(
            root=data_dir, domain=args.source, 
            split='train', transform=train_transform
            )
        validsets = {}
        for dd in domains:
            dd_validset = pacs.PACS(
                root=data_dir, domain=dd,
                split='crossval', transform=test_transform
                )
            validsets[dd] = dd_validset
        testsets={}
        for dd in trg_domains:
            dd_testset = pacs.PACS(
                root=data_dir, domain=dd,
                split='test', transform=test_transform
                )
            testsets[dd] = dd_testset
        # add the source crossval as a test too
        testsets[args.source] = validsets[args.source]
    
    elif args.data_name=='officehome':
        sourceset = ImageFolder(
            os.path.join(data_dir, args.source), 
            transform=train_transform
            )
        trainset, src_validset = torch.utils.data.random_split(
            sourceset, [
                int(0.9*len(sourceset)), 
                len(sourceset) - int(0.9*len(sourceset))
                ], 
            generator=torch.Generator().manual_seed(381)
            )
        validsets = {} 
        testsets = {}
        for dd in trg_domains:
            dd_set = ImageFolder(
                os.path.join(data_dir, dd), transform=test_transform) 
            dd_validset, dd_testset = torch.utils.data.random_split(
                dd_set, 
                [int(0.1*len(dd_set)), len(dd_set) - int(0.1*len(dd_set))], 
                generator=torch.Generator().manual_seed(381)
                )
            validsets[dd] = dd_validset 
            testsets[dd] = dd_testset
        validsets[args.source] = src_validset 
        testsets[args.source] = src_validset

    elif args.data_name=='digits':
        trainset = get_dataset(
            args.source, root=data_dir, train=True, download=True, 
            transform=train_transform
            )
        validsets = {
            domain: get_dataset(
                domain, root=data_dir, train=False, download=True, transform=test_transform) for domain in domains}

        testsets = validsets

    trainloaders = [
        torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, 
            num_workers=8
            )
        ]
    validloaders = {
        d: torch.utils.data.DataLoader(
            validsets[d], batch_size=args.batch_size, shuffle=False, 
            num_workers=2
            ) for d in validsets.keys()
        }
    testloaders = {
        d: torch.utils.data.DataLoader(
            testsets[d], batch_size=args.batch_size, shuffle=False, 
            num_workers=2
            ) for d in testsets.keys()
        }

    # MODEL
    print("\n=========Building Model=========")
    net = Learner(
        get_network(
            args.net, num_classes=args.n_classes, pretrained=True, 
            drop=args.drop
            )
        )
    trainer = ALT(args)
    trainer.train(
        net, 
        trainset,
        trainloaders, validloaders, testloaders=testloaders, 
        data_mean=(0.5, 0.5, 0.5), data_std=((0.5, 0.5, 0.5))
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    add_basic_args(parser)
    args = parser.parse_args()
    main(args)