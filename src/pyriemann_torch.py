import torch
import numpy
import scipy
from numpy.core.numerictypes import typecodes


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
            tmp = torch.mm(torch.mm(Cm12, covmats[index, :, :]), Cm12)
            J += sample_weight[index] * logm(tmp)

        crit = numpy.linalg.norm(J, ord='fro')
        h = nu * crit
        C = torch.mm(torch.mm(C12, expm(nu * J)), C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C

def _matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
    if Ci.numpy().dtype.char in typecodes['AllFloat'] and not torch.isfinite(Ci).all():
        raise ValueError("Covariance matrices must be positive definite. Add regularization to avoid this error.")
    eigvals, eigvects = scipy.linalg.eigh(Ci, check_finite=False)
    eigvects = torch.FloatTensor(eigvects)
    eigvals = torch.diag(operator(torch.FloatTensor(eigvals)))
    Out = torch.mm(torch.mm(eigvects, eigvals), eigvects.T)
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


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def computeCOV(X):
    m = torch.arange(10, dtype=torch.float64)
    f = torch.arange(10) * 2
    a = torch.arange(10) ** 2.
    ddof = 9  # N - 1
    w = f * a
    v1 = torch.sum(w)
    v2 = torch.sum(w * a)
    m -= torch.sum(m * w, axis=None, keepdims=True) / v1
    cov = torch.mm(m * w, m.T) * v1 / (v1 ** 2 - ddof * v2)

def covariances(X):
    """Estimation of covariance matrix."""
    Nt, Ns, Ne = X.shape
    covmats = torch.zeros((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i, :, :] = cov(X[i, :, :])
    return covmats

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

def tangent_space(covmats, Cref):
    """Project a set of covariance matrices in the tangent space. according to
    the reference point Cref

    :param covmats: np.ndarray
        Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param Cref: np.ndarray
        The reference covariance matrix
    :returns: np.ndarray
        the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)

    """
    Nt, Ne, Ne = covmats.shape
    Cm12 = invsqrtm(Cref)
    idx = numpy.triu_indices_from(Cref)
    Nf = int(Ne * (Ne + 1) / 2)
    T = torch.empty((Nt, Nf))
    coeffs = (numpy.sqrt(2) * numpy.triu(numpy.ones((Ne, Ne)), 1) +
              numpy.eye(Ne))[idx]
    for index in range(Nt):
        tmp = torch.mm(torch.mm(Cm12, covmats[index, :, :]), Cm12)
        tmp = logm(tmp)
        T[index, :] = torch.mul(torch.FloatTensor(coeffs), tmp[idx])
    return T


def untangent_space(T, Cref):
    """Project a set of Tangent space vectors back to the manifold.

    :param T: np.ndarray
        the Tangent space , a matrix of Ntrials X (channels * (channels + 1)/2)
    :param Cref: np.ndarray
        The reference covariance matrix
    :returns: np.ndarray
        A set of Covariance matrix, Ntrials X Nchannels X Nchannels

    """
    Nt, Nd = T.shape
    Ne = int((torch.sqrt(1 + 8 * Nd) - 1) / 2)
    C12 = sqrtm(Cref)

    idx = torch.triu_indices_from(Cref)
    covmats = torch.empty((Nt, Ne, Ne))
    covmats[:, idx[0], idx[1]] = T
    for i in range(Nt):
        triuc = torch.triu(covmats[i], 1) / torch.sqrt(2)
        covmats[i] = (torch.diag(torch.diag(covmats[i])) + triuc + triuc.T)
        covmats[i] = expm(covmats[i])
        covmats[i] = torch.mm(torch.mm(C12, covmats[i]), C12)

    return covmats


def transport(Covs, Cref):
    """Parallel transport of two set of covariance matrix.

    """
    C = mean_riemann(Covs)
    iC = invsqrtm(C)
    E = sqrtm(torch.mm(torch.mm(iC, Cref), iC))
    out = torch.array([torch.mm(torch.mm(E, c), E.T) for c in Covs])
    return out
