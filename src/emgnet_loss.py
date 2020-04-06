# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class FeLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(FeLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return emg_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def emg_loss(input, target, n_support, test_flag = None):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    y = target.to('cpu')
    x = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return y.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(y)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = y.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    # FIXME sd
    prototypes = torch.stack([x[idx_list].mean(0) for idx_list in support_idxs]) # support vector의 평균

    if test_flag == None:
        n_query2use = n_support * 2
        euclidean_dist_list = torch.FloatTensor(n_classes, n_query2use, prototypes.shape[0])
        for c in classes:
            query_samples = input.to('cpu')[y.eq(c).nonzero()[n_support:].view(-1)]
            dists= euclidean_dist(query_samples, prototypes)
            for i_proto in range(prototypes.shape[0]):
                distSorted, idxSorted = dists[:, i_proto].sort()
                euclidean_dist_list[c, :, i_proto] = distSorted[:n_query2use]
        euclidean_dist_list = euclidean_dist_list.view(-1, prototypes.shape[0])
        log_p_y = F.log_softmax(-euclidean_dist_list, dim=1).view(n_classes, n_query2use, -1)

        target_inds = torch.arange(0, n_classes)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query2use, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    else:
        query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[n_support:], classes))).view(-1)
        query_samples = input.to('cpu')[query_idxs]

        # distance between samples in query samples, prototypes
        dists = euclidean_dist(query_samples, prototypes)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

        target_inds = torch.arange(0, n_classes)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val, acc_val, y_hat
