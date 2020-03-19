# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import scipy
from numpy.core.numerictypes import typecodes
import numpy
import pyriemann

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def prototypical_loss(input, target, n_support):
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
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    # FIXME get supprots using Riemannian mean
    # prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs]) # support vector의 평균
    # riemannain mean
    for idx_list in support_idxs:
        mean_riemann(input_cpu[idx_list])

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])  # support vector의 평균

    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

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

    return loss_val,  acc_val

def mean_riemann(covmats, tol=10e-9, maxiter=50, init=None,
                 sample_weight=None):
    """Return the mean covariance matrix according to the Riemannian metric.

    The procedure is similar to a gradient descent minimizing the sum of
    riemannian distance to the mean.

    .. math::
            \mathbf{C} = \\arg\min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)}  # noqa

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the gradient descent. If None the Arithmetic mean is used
    :param sample_weight: the weight of each sample
    :returns: the mean covariance matrix

    """
    # init
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = torch.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    nu = 1.0
    tau = torch.finfo(torch.float64).max
    crit = torch.finfo(torch.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C)
        Cm12 = invsqrtm(C)
        J = torch.zeros((Ne, Ne))

        for index in range(Nt):
            tmp = torch.dot(torch.dot(Cm12, covmats[index, :, :]), Cm12)
            J += sample_weight[index] * logm(tmp)

        crit = torch.linalg.norm(J, ord='fro')
        h = nu * crit
        C = torch.dot(torch.dot(C12, expm(nu * J)), C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C

def _get_sample_weight(sample_weight, data):
    """Get the sample weights.

    If none provided, weights init to 1. otherwise, weights are normalized.
    """
    if sample_weight is None:
        sample_weight = torch.ones(data.shape[0])
    if len(sample_weight) != data.shape[0]:
        raise ValueError("len of sample_weight must be equal to len of data.")
    sample_weight /= torch.sum(sample_weight)
    return sample_weight

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


def _matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
    if Ci.dtype.char in typecodes['AllFloat'] and not torch.isfinite(Ci).all():
        raise ValueError("Covariance matrices must be positive definite. Add regularization to avoid this error.")
    eigvals, eigvects = scipy.linalg.eigh(Ci, check_finite=False)
    eigvals = torch.diag(operator(eigvals))
    Out = torch.dot(torch.dot(eigvects, eigvals), eigvects.T)
    return Out


def sqrtm(Ci):
    """Return the matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix square root

    """
    return _matrix_operator(Ci, torch.sqrt)


def logm(Ci):
    """Return the matrix logarithm of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix logarithm

    """
    return _matrix_operator(Ci, torch.log)


def expm(Ci):
    """Return the matrix exponential of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix exponential

    """
    return _matrix_operator(Ci, torch.exp)


def invsqrtm(Ci):
    """Return the inverse matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the inverse matrix square root

    """
    isqrt = lambda x: 1. / torch.sqrt(x)
    return _matrix_operator(Ci, isqrt)


def powm(Ci, alpha):
    """Return the matrix power :math:`\\alpha` of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{\\alpha} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :param alpha: the power to apply
    :returns: the matrix power

    """
    power = lambda x: x**alpha
    return _matrix_operator(Ci, power)


def _matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
    # Fixme "Ci.dtype.char in typecodes['AllFloat'] and" is exlcuded, which may be ported later.
    if not torch.isfinite(Ci).all():
        raise ValueError("Covariance matrices must be positive definite. "
                         "Add regularization to avoid this error.")
    # eigvals, eigvects = scipy.linalg.eigh(Ci, check_finite=False)
    eigvals, eigvects = torch.symeig(Ci, eigenvectors=True)
    torch.sqrt
    eigvals = torch.diag(operator(eigvals))

    # Out = torch.dot(torch.dot(eigvects, eigvals), eigvects.T)
    Out = torch.mm(torch.mm(eigvects, eigvals.unsqueeze(1)),eigvects.T).squeeze()
    return Out
