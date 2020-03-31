import torch
import numpy

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
    # sample_weight = _get_sample_weight(sample_weight, covmats)
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
    cov = torch.dot(m * w, m.T) * v1 / (v1 ** 2 - ddof * v2)

def covariances(X):
    """Estimation of covariance matrix."""
    Nt, Ns, Ne = X.shape
    covmats = torch.zeros((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i, :, :] = cov(X[i, :, :])
    return covmats
