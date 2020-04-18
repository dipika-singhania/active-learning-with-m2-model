import torch
import random
import numpy as np


class AdversarySampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, vae, discriminator, data, cuda):
        all_preds = []
        all_indices = []
        labels = []
        for images, label, indices in data:
            if vae is not None and discriminator is not None:
                if cuda:
                    images = images.cuda()

                with torch.no_grad():
                    pred_images = vae.classify(images)
                    _, _, mu, _ = vae(images, pred_images)
                    preds = discriminator(mu, pred_images)

                preds = preds.cpu().data
                all_preds.extend(preds)
                labels.extend(label)
            all_indices.extend(indices)

        if vae is not None and discriminator is not None:       
            all_preds = torch.stack(all_preds)
            all_preds = all_preds.view(-1)
            # need to multiply by -1 to be able to use torch.topk 
            all_preds *= -1

            # select the points which the discriminator things are the most likely to be unlabeled
            _, querry_indices = torch.topk(all_preds, int(self.budget))
            querry_pool_indices = np.asarray(all_indices)[querry_indices]
            labels_chosen = np.asarray(labels)[querry_indices]
            print("Labels Choosen are ", np.unique(labels_chosen, return_counts=True))
            return querry_pool_indices
        else:
            all_indices = set(np.asarray(all_indices))
            query_indices = random.sample(all_indices, int(self.budget))
            return query_indices
