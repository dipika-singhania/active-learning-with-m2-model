import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler
import copy
from LRFinder_Discrim import *
from LRFinderVae import *
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torch.autograd import Variable
import sys


def print_tensorboard_results(epoch_writer, iteration, amt_data, dict_params):
    label_base = 'amt_data_' + '{:5d}'.format(amt_data)
    epoch_writer.add_scalars(label_base, dict_params, iteration)


def load_checkpoint(model_file_path, p_name, p_vae, p_discriminator, p_vae_optim,
                    p_discriminator_optim, is_best=False):
    if is_best:
        file_name = os.path.join(model_file_path, p_name + '_best.pth.tar')
        chk = torch.load(file_name)
        print("Picking up best model with name = ", file_name)
    else:
        file_name = os.path.join(model_file_path, p_name + '.pth.tar')
        chk = torch.load(file_name)
        print("Picking up model with name = ", file_name)
    l_iter = chk['iter']
    l_test_perf = chk['test_perf']
    l_val_perf = chk['val_perf']

    print("Epoch picked up from = ", l_iter, "test performance of model =", l_test_perf, "val last performance", l_val_perf)
    # p_task_model.load_state_dict(chk['task_model_state_dict'])
    p_vae.load_state_dict(chk['vae_model_state_dict'])
    p_discriminator.load_state_dict(chk['discrimnator_model_state_dict'])
    p_vae_optim.load_state_dict(chk['vae_optim_state_dict'])
    p_discriminator_optim.load_state_dict(chk['discriminator_optim_state_dict'])

    return l_iter, l_test_perf, l_val_perf


def save_model(model_file_path, p_name, p_vae, p_discriminator, p_vae_optim,
                p_discriminator_optim, p_iter, p_test_perf, p_val_perf, is_best=False):

    # 'task_model_state_dict': p_task_model.state_dict(),
    torch.save({'vae_model_state_dict': p_vae.state_dict(),
                'discrimnator_model_state_dict': p_discriminator.state_dict(),
                'vae_optim_state_dict': p_vae_optim.state_dict(),
                'discriminator_optim_state_dict': p_discriminator_optim.state_dict(),
                'iter': p_iter, 'test_perf': p_test_perf, 'val_perf': p_val_perf},
                 os.path.join(model_file_path, p_name + '.pth.tar'))
    if is_best:
        # 'task_model_state_dict': p_task_model.state_dict()
        torch.save({'vae_model_state_dict': p_vae.state_dict(),
                    'discrimnator_model_state_dict': p_discriminator.state_dict(),
                    'vae_optim_state_dict': p_vae_optim.state_dict(),
                    'discriminator_optim_state_dict': p_discriminator_optim.state_dict(),
                    'iter': p_iter, 'test_perf': p_test_perf, 'val_perf': p_val_perf},
                    os.path.join(model_file_path, p_name + '_best.pth.tar'))


class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader
        self.model_path = self.args.out_path
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        # self.base_name = "discriminator_probs_"    # Only in the discriminator probabilties changes, no backward vae
        # self.base_name = "vae_back_dis_probs_"    # Vae backward done and discriminator probabilities only
        self.base_name = "m2_model"
        self.sampler = sampler.AdversarySampler(self.args.budget)
        if args.tensorboard is True:
            tb_path = os.path.join(args.out_path, 'tb_logs')
            if not os.path.exists(tb_path):
                os.mkdir(tb_path)
            self.writer_train = SummaryWriter(os.path.join(tb_path, 'summary_train'))
            self.writer_val = SummaryWriter(os.path.join(tb_path, 'summary_val'))

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    def lr_finder_vae(self, querry_dataloader, unlabeled_dataloader, vae, discriminator, optim_vae):
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)
        vae.train()
        discriminator.train()

        if self.args.cuda:
            vae = vae.cuda()

        device = torch.device('cuda:0') if self.args.cuda else torch.device('cpu')
        lr_finder = LRFinder(model=vae, optimizer=optim_vae, criterion=[self.ce_loss, self.mse_loss],\
                             device=device)
        lr_finder.range_test([labeled_data, unlabeled_data], end_lr=10, num_iter=200, step_mode="exp")
        lr_finder.plot(fname='lr_probing_vae' + self.args.dataset + '.pdf')
        sys.exit(1)

    def lr_finder_ad(self, querry_dataloader, unlabeled_dataloader, vae, discriminator):
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-8)

        # l_name = 'm2_model_20.0'
        if self.args.cuda:
            vae = vae.cuda()
            discriminator = discriminator.cuda()

        vae.eval()
        discriminator.train()
        # load_checkpoint(self.model_path, l_name, vae, discriminator, optim_vae, optim_discriminator, True)

        device = torch.device('cuda:0') if self.args.cuda else torch.device('cpu')
        lr_finder = LRFinderDiscrim(model=[vae, discriminator], optimizer=optim_discriminator, criterion=self.bce_loss,\
                             device=device)
        lr_finder.range_test([labeled_data, unlabeled_data], end_lr=10, num_iter=200, step_mode="exp")
        lr_finder.plot(fname='lr_probing_discriminator' + self.args.dataset + '.pdf')
        sys.exit(1)

    def train(self, querry_dataloader, val_dataloader, vae, discriminator, optim_vae, optim_discriminator,
              unlabeled_dataloader, size_of_labeled_data, p_resume):
        self.args.train_iterations = (self.args.num_images * self.args.train_epochs) // self.args.batch_size
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)
        l_name = self.base_name + "_" + "{:5d}".format(size_of_labeled_data)

        vae.train()
        discriminator.train()

        if self.args.cuda:
            vae = vae.cuda()
            discriminator = discriminator.cuda()


        l_iter = 0
        if p_resume is True:
            l_iter, l_test_perf, l_val_perf = load_checkpoint(self.model_path, l_name, vae, discriminator,
                                                              optim_vae, optim_discriminator, False)
        best_acc = 0
        is_best = False
        for iter_count in range(l_iter, self.args.train_iterations):

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()


            # VAE step
            for count in range(self.args.num_vae_steps):
                preds_labelled = vae.classify(labeled_imgs)
                preds_unlabelled = vae.classify(unlabeled_imgs)
                task_loss = self.ce_loss(preds_labelled, labels)

                recon, z, mu, logvar = vae(labeled_imgs, preds_labelled)
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta)

                unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs, preds_unlabelled)
                transductive_loss = self.vae_loss(unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, self.args.beta)

                labeled_preds = discriminator(mu, torch.log(preds_labelled + 1e-15))
                unlabeled_preds = discriminator(unlab_mu, torch.log(preds_unlabelled + 1e-15))
                
                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                    
                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_real_preds = unlab_real_preds.cuda()

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                           self.bce_loss(unlabeled_preds, unlab_real_preds)

                total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss + task_loss
                # total_vae_loss = unsup_loss + transductive_loss + task_loss
                # total_vae_loss = task_loss
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()


            # Discriminator step
            for count in range(self.args.num_adv_steps):
                with torch.no_grad():
                    discrim_pred_labelled = vae.classify(labeled_imgs)
                    discrim_pred_unlabelled = vae.classify(unlabeled_imgs)
                    _, _, mu, _ = vae(labeled_imgs, discrim_pred_labelled)
                    _, _, unlab_mu, _ = vae(unlabeled_imgs, discrim_pred_unlabelled)
                
                labeled_preds = discriminator(mu, torch.log(discrim_pred_labelled + 1e-15))
                unlabeled_preds = discriminator(unlab_mu, torch.log(discrim_pred_unlabelled + 1e-15))
                
                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_fake_preds = unlab_fake_preds.cuda()
                
                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                           self.bce_loss(unlabeled_preds, unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

            if iter_count % 100 == 0:
                if self.args.tensorboard:
                    print_tensorboard_results(self.writer_train, iter_count, size_of_labeled_data, {
                        'task_loss': task_loss.item(),
                        'total_vae_loss': total_vae_loss.item(),
                        'dcs_loss': dsc_loss.item()
                    })
                print('Current training iteration: {}'.format(iter_count))
                print('Current task model loss: {:.4f}'.format(task_loss.item()))
                print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))

            if iter_count % 100 == 0:
                acc = self.validate(vae, val_dataloader)
                if self.args.tensorboard:
                    print_tensorboard_results(self.writer_val, iter_count, size_of_labeled_data, {'acc': acc})
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(vae)
                    is_best = True
                else:
                    is_best = False

                print('current step: {:2d} acc: {:2.2f}'.format(iter_count, acc))

                print('best validation acc: {:2.2f}'.format(best_acc))
                
            if iter_count % 1000 == 0:
                final_accuracy = self.test(best_model)
                print('Completed {:5.2f} epochs, test best acc using {:5d} data is: {:2.2f}'.format((iter_count / self.args.num_images),
                                                                                  size_of_labeled_data, final_accuracy))
                print('best validation acc: ', best_acc)
                if not self.args.test_acc_only:
                    save_model(self.model_path, l_name, vae, discriminator, optim_vae,
                               optim_discriminator, iter_count, final_accuracy, acc, is_best)
            
        if not self.args.test_acc_only:
            best_model = best_model.cuda()
            save_model(self.model_path, l_name, best_model, discriminator, optim_vae,
                       optim_discriminator, iter_count, final_accuracy, acc, True)

        final_accuracy = self.test(best_model)

        return final_accuracy

    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader):
        querry_indices = self.sampler.sample(vae, discriminator, unlabeled_dataloader,
                                             self.args.cuda)

        return querry_indices

    def random_sampling(self, unlabeled_dataloader):
        querry_indices = self.sampler.sample(None, None, unlabeled_dataloader,
                                             self.args.cuda)

        return querry_indices

    def load_and_test(self, vae, discriminator, optim_vae, optim_discriminator, size_of_labeled_data):
        # optim_task_model = optim.SGD(task_model.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9)
        l_name = self.base_name + "_" + "{:5d}".format(size_of_labeled_data)
        if self.args.cuda:
            vae = vae.cuda()
            discriminator = discriminator.cuda()
            # task_model = task_model.cuda()

        l_iter, l_test_perf, l_val_perf = load_checkpoint(self.model_path, l_name, vae, discriminator,
                                                          optim_vae, optim_discriminator, False)
        # final_accuracy = self.test(task_model)
        final_accuracy = self.test(vae)
        print('Loaded model with iter {:4d} epochs, test acc reported {:2.2f}, val acc reported {:2.2f} ,'
              'got actual test acc with current model using {:5d} data is:{:2.2f}'.format(l_iter, l_test_perf,
                                                                                          l_val_perf,
                                                                                          size_of_labeled_data,
                                                                                          final_accuracy))
        return final_accuracy

    def load_and_see_few_iamges(self, vae, discriminator, optim_vae, optim_discriminator, size_of_labeled_data):
        # optim_task_model = optim.SGD(task_model.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9)
        l_name = self.base_name + "_" + "{:5d}".format(size_of_labeled_data)
        if self.args.cuda:
            vae = vae.cuda()
            vae.eval()
            discriminator = discriminator.cuda()
            discriminator.eval()
            # task_model = task_model.cuda()

        l_iter, l_test_perf, l_val_perf = load_checkpoint(self.model_path, l_name, vae, discriminator,
                                                           optim_vae, optim_discriminator, True)
        imgs, label = next(iter(self.test_dataloader))

        if self.args.cuda:
            imgs = imgs.cuda()

        with torch.no_grad():
            preds = vae.classify(imgs)
            z = Variable(torch.randn(imgs.shape[0], 32))
            z = z.cuda()
            x_mu = vae.sample(z, preds)
            f, axarr = plt.subplots(2, 16, figsize=(18, 12))

            samples = x_mu.data.view(-1, 28, 28).cpu().numpy()
            predictions = np.argmax(preds.cpu().numpy(), axis=-1)
            data_dict = {0: 'T - shirt / top', 1: 'Trouser', 2:'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal',
                         6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
            for i, ax in enumerate(axarr.flat):
                plt.subplot(2, 16, i + 1)
                plt.imshow(samples[i])
                plt.axis("off")
                plt.title(data_dict[predictions[i]])
                plt.subplot(2, 16, 17 + i)
                plt.imshow(imgs[i].view(28, 28).cpu().numpy())
                plt.axis("off")
                plt.title(data_dict[label[i].item()])
                if i >= 15:
                    break
            f.savefig(os.path.join(self.args.out_path, 'image_' + "{:5d}".format(size_of_labeled_data) + ".pdf"))

    def validate(self, task_model, loader):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels, _ in loader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model.classify(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        task_model.train()
        return correct / total * 100

    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model.classify(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        task_model.train()
        return correct / total * 100

    def vae_loss(self, x, recon, mu, logvar, beta):
        # x = x.view(-1, recon.shape[1])
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
