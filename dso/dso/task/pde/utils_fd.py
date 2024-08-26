import numpy as np

from dso.task.pde.utils_v2 import central_diff
from dso.task.pde.utils_v1 import FiniteDiff2


def Diff(u, dxt, dim, name='x'):
    ##############################
    # This function is used to compute the central difference of an array along axis 0
    # Args:
    #     u (np.array): the array to be differentiated
    #     dx (float): the step size
    #     dim (int, to be deprecated): the axis along which to differentiate
    #     name (str, to be deprecated): the name of the variable to differentiate
    # Returns:
    #     np.array: the differentiated array
    ##############################
    dxt = dxt[2] - dxt[1]
    uxt = central_diff(u, dxt, axis=0)

    return uxt


def Diff2(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    dxt = dxt[2] - dxt[1]
    for i in range(m):
        uxt[:, i] = FiniteDiff2(u[:, i], dxt)

    return uxt


def Diff3(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    dxt = dxt[2] - dxt[1]
    uxt = central_diff(u, dxt, axis=0)
    for i in range(m):
        uxt[:, i] = FiniteDiff2(uxt[:, i], dxt)
    return uxt


def Diff4(u, dxt, dim, name='x'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    n, m = u.shape
    uxt = np.zeros((n, m))

    if name == 'x':
        dxt = dxt[2] - dxt[1]
        for i in range(m):
            uxt[:, i] = FiniteDiff2(u[:, i], dxt)
            uxt[:, i] = FiniteDiff2(uxt[:, i], dxt)
    elif name == 't':
        for i in range(n):
            uxt[i, :] = FiniteDiff2(u[i, :], dxt)
            uxt[i, :] = FiniteDiff2(uxt[:, i], dxt)

    else:
        NotImplementedError()
