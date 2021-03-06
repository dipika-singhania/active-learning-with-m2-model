import torch
from torchvision import datasets, transforms
import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
import vgg
from solver import Solver
import arguments
import torch.utils.data.sampler  as sampler
import torch.utils.data as data
import sys
import torch.optim as optim

def main(args):
    args.cuda = True if torch.cuda.is_available() else False
    if args.log_file is not None and len(args.log_file) > 0:
        sys.stdout = open(args.log_file, 'w')
    if args.dataset == 'MNIST':
        test_dataloader = data.DataLoader(
            datasets.MNIST(args.data_path, download=True, transform=mnist_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)

        train_dataset = MNIST(args.data_path)
        print("Shape of one image = ", train_dataset.__getitem__(0)[0].shape)
        args.num_images = 60000
        args.num_val = 5000
        args.budget = 20
        args.initial_budget = 20
        args.num_classes = 10
        x_dim = 784
        h_dim = [256, 128]
        args.lr_vae = 3e-3
        args.lr_ad = 3e-2

    elif args.dataset == 'FashionMNIST':
        test_dataloader = data.DataLoader(
            datasets.FashionMNIST(args.data_path, download=True, transform=mnist_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)

        train_dataset = FashionMnist(args.data_path)
        print("Shape of one image = ", train_dataset.__getitem__(0)[0].shape)
        args.num_images = 60000
        args.num_val = 5000
        args.budget = 64
        args.initial_budget = 500
        args.num_classes = 10
        x_dim = [1, 28, 28]
        h_dim = [64, 32, 512]
        args.lr_vae = 3e-4
        args.lr_ad = 3e-4

    elif args.dataset == 'cifar10':
        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR10(args.data_path)
        args.num_images = 50000
        args.num_val = 5000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 10
        x_dim = 3072
        h_dim = [1024, 512, 256, 128]
        args.lr_vae = 3e-4
        args.lr_ad = 3e-4

    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR100(args.data_path)

        args.num_val = 5000
        args.num_images = 50000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 100
        x_dim = 3 * 32 * 32

    elif args.dataset == 'imagenet':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = ImageNet(args.data_path)

        args.num_val = 128120
        args.num_images = 1281167
        args.budget = 64060
        args.initial_budget = 128120
        args.num_classes = 1000
    else:
        raise NotImplementedError

    random.seed(1234)
    args.out_path = os.path.join(args.out_path, args.dataset)
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    output_file_name = os.path.join(args.out_path, 'final_accuracies.log')
    all_indices = set(np.arange(args.num_images))

    #  Sampling from few classes only
    #  Taking initial samples to be [2, 4, 7]

    count = 0
    initial_indices_list = []
    target_indices = []
    for indices in all_indices:
        _, target, index = train_dataset.__getitem__(indices)
        if target in [2, 4, 7]:
            initial_indices_list.append(index)
            count += 1
            target_indices.append(target)
        if count == args.initial_budget:
            break

    print("Initial values of data sampled", np.unique(target_indices))

    val_indices = random.sample(all_indices, args.num_val)
    all_indices = np.setdiff1d(list(all_indices), val_indices)

    initial_indices = random.sample(list(initial_indices_list), args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                                        batch_size=args.batch_size, drop_last=True)
    val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler,
                                     batch_size=args.batch_size, drop_last=False)
            
    args.cuda = args.cuda and torch.cuda.is_available()
    solver = Solver(args, test_dataloader)

    splits = []
    images_used = []
    for ele in range(args.initial_budget, int(0.20 * args.num_images),  args.budget):
        splits.append(float(ele)/args.num_images)
        images_used.append(ele)
    # print("Splits used is:", splits)
    print("Splits number of images:", images_used)
    # splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    current_indices = list(initial_indices)

    accuracies = []
    start = 0
    if args.resume is True:
        start = args.start_resume

    if args.m1_model:
        vae = model.VariationalAutoEncoder([x_dim, args.num_classes, args.latent_dim, h_dim], args.dataset)
    else:
        vae = model.DeepGenerativeModel([x_dim, args.num_classes, args.latent_dim, h_dim], args.dataset)

    optim_vae = optim.Adam(vae.parameters(), lr=args.lr_vae)

    for split in images_used[start:]:
        # need to retrain all the models on the new images
        # re initialize and retrain the models
        # task_model = vgg.vgg16_bn(num_classes=args.num_classes)
        # vae = model.VAE(z_dim=args.latent_dim, nc=3, class_probs=args.num_classes)
        # task_model = None
        if args.m1_model:
            discriminator = model.Discriminator(z_dim=args.latent_dim, class_probs=0)
        else:
            discriminator = model.Discriminator(z_dim=args.latent_dim, class_probs=args.num_classes)

        optim_discriminator = optim.Adam(discriminator.parameters(), lr=args.lr_ad)

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset, sampler=unlabeled_sampler,
                                               batch_size=args.batch_size, drop_last=False)

        if args.visualization:
            solver.load_and_see_few_iamges(vae, discriminator, optim_vae, optim_discriminator, split)
            acc = 0
        elif args.find_lr_vae:
            solver.lr_finder_vae(querry_dataloader, unlabeled_dataloader, vae, discriminator, optim_vae,
                                 optim_discriminator)
            acc = 0
        elif args.find_lr_ad:
            solver.lr_finder_ad(querry_dataloader, unlabeled_dataloader, vae, discriminator, optim_vae,
                                optim_discriminator)
            acc = 0
        elif args.test_acc_only:
            acc = solver.load_and_test(vae, discriminator, optim_vae, optim_discriminator, split)
        else:
            # train the models on the current data
            acc = solver.train(querry_dataloader,
                               val_dataloader,
                               vae,
                               discriminator,
                               optim_vae,
                               optim_discriminator,
                               unlabeled_dataloader,
                               split, p_resume=args.resume)
        args.resume = False
        print("Final accuracy with ", split, " images of data is: {:2.2f}".format(acc))
        if not args.test_acc_only:
            with open(output_file_name, 'a+') as fp:
                fp.write("Final accuracy with " + str(split) + " images of data is: " + "{:2.2f}".format(acc) + "\n")
        accuracies.append(acc)

        if args.random_sampling is True:
            sampled_indices = solver.random_sampling(unlabeled_dataloader)
        else:
            sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, drop_last=True)


if __name__ == '__main__':
    args = arguments.get_args()
    main(args)

