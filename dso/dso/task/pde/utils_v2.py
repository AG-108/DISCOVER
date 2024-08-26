from logging import warning

import numpy as np

from numba import jit, njit


def array_slice(a, axis, start, end, step=1):
    ##############################
    # This function is used to slice an array along a given axis
    # Args:
    #     a (np.array): the array to be sliced
    #     axis (int): the axis along which to slice
    #     start (int): the starting index
    #     end (int): the ending index
    #     step (int): the step size
    # Returns:
    #     np.array: the sliced array
    ##############################
    ret = a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]
    return ret


def central_diff(u, dx, axis=0):
    ##############################
    # This function is used to compute the central difference of an array along a given axis
    # Args:
    #     u (np.array): the array to be differentiated
    #     dx (float): the step size
    #     axis (int): the axis along which to differentiate
    # Returns:
    #     np.array: the differentiated array
    ##############################
    n = u.shape[axis]
    ux = np.zeros_like(u)
    ux[(slice(None),) * (axis % ux.ndim) + (slice(1, n - 1),) + (slice(None),) * (ux.ndim - 1 - axis % ux.ndim)] \
        = (array_slice(u, axis, 2, n) - array_slice(u, axis, 0, n - 2)) / (2 * dx)
    ux[(slice(None),) * (axis % ux.ndim) + (slice(0, 1),) + (slice(None),) * (ux.ndim - 1 - axis % ux.ndim)] \
        = (-3.0 / 2 * array_slice(u, axis, 0, 1)
           + 2 * array_slice(u, axis, 1, 2)
           - array_slice(u, axis, 2, 3) / 2) / dx
    ux[(slice(None),) * (axis % ux.ndim) + (slice(n - 1, n),) + (slice(None),) * (ux.ndim - 1 - axis % ux.ndim)] \
        = (3.0 / 2 * array_slice(u, axis, n - 1, n)
           - 2 * array_slice(u, axis, n - 2, n - 1)
           + array_slice(u, axis, n - 3, n - 2) / 2) / dx
    return ux


def forward_diff(u, dx, axis=0):
    ##############################
    # This function is used to compute the forward difference of an array along a given axis
    # Args:
    #     u (np.array): the array to be differentiated
    #     dx (float): the step size
    #     axis (int): the axis along which to differentiate
    # Returns:
    #     np.array: the differentiated array
    ##############################
    n = u.shape[axis]
    ux = np.zeros_like(u)
    ux[(slice(None),) * (axis % ux.ndim) + (slice(1, n),) + (slice(None),) * (ux.ndim - 1 - axis % ux.ndim)] \
        = (array_slice(u, axis, 1, n) - array_slice(u, axis, 0, n - 1)) / dx
    ux[(slice(None),) * (axis % ux.ndim) + (slice(0, 1),) + (slice(None),) * (ux.ndim - 1 - axis % ux.ndim)] \
        = (array_slice(u, axis, 1, 2) - array_slice(u, axis, 0, 1)) / dx
    return ux


def backward_diff(u, dx, axis=0):
    ##############################
    # This function is used to compute the backward difference of an array along a given axis
    # Args:
    #     u (np.array): the array to be differentiated
    #     dx (float): the step size
    #     axis (int): the axis along which to differentiate
    # Returns:
    #     np.array: the differentiated array
    ##############################
    n = u.shape[axis]
    ux = np.zeros_like(u)
    ux[(slice(None),) * (axis % ux.ndim) + (slice(0, n - 1),) + (slice(None),) * (ux.ndim - 1 - axis % ux.ndim)] \
        = (array_slice(u, axis, 1, n) - array_slice(u, axis, 0, n - 1)) / dx
    ux[(slice(None),) * (axis % ux.ndim) + (slice(n - 1, n),) + (slice(None),) * (ux.ndim - 1 - axis % ux.ndim)] \
        = (array_slice(u, axis, n - 1, n) - array_slice(u, axis, n - 2, n - 1)) / dx
    return ux


# @jit(nopython=True)
def Diff_2(u, dxt, name=1, method='central'):
    """
    Here dx is a scalar, name is a str indicating what it is
    """
    # import pdb;pdb.set_trace()ter
    if u.shape == dxt.shape:
        return u / dxt
    if name > 2:
        assert False, f'Axis {name} out of range'
    if len(dxt.shape) == 2:
        dxt = dxt[:, 0]
    dxt = dxt.ravel()
    if u.shape[name] == 2:
        method = 'forward'
        warning(f'Axis {name} is too short, switch to forward method')
    # import pdb;pdb.set_trace()
    dxt = np.mean(np.diff(dxt))
    if method == 'central':
        uxt = central_diff(u, dxt, axis=name)
    elif method == 'forward':
        uxt = forward_diff(u, dxt, axis=name)
    elif method == 'backward':
        uxt = backward_diff(u, dxt, axis=name)
    else:
        assert False, f'{method} is not a supported differentiation method'

    return uxt


# @jit(nopython=True)
def Diff2_2(u, dxt, name=1):
    """
    Here dx is a scalar, name is a str indicating what it is
    """

    if u.shape == dxt.shape:
        return u / dxt
    t, n, m = u.shape
    uxt = np.zeros((t, n, m))
    dxt = dxt.ravel()
    # try: 
    if name == 1:
        dxt = dxt[2] - dxt[1]
        uxt[:, 1:n - 1, :] = (u[:, 2:n, :] - 2 * u[:, 1:n - 1, :] + u[:, 0:n - 2, :]) / dxt ** 2
        uxt[:, 0, :] = (u[:, 1, :] + u[:, -1, :] - 2 * u[:, 0, :]) / dxt ** 2
        uxt[:, -1, :] = (u[:, 0, :] + u[:, -2, :] - 2 * u[:, -1, :]) / dxt ** 2
        # uxt[:,0,:] = (2 * u[:,0,:] - 5 * u[:,1,:] + 4 * u[:,2,:] - u[:,3,:]) / dxt ** 2
        # uxt[:,n - 1,:] = (2 * u[:,n - 1,:] - 5 * u[:,n - 2,:] + 4 * u[:,n - 3,:] - u[:,n - 4,:]) / dxt ** 2
    elif name == 2:
        dxt = dxt[2] - dxt[1]
        uxt[:, :, 1:m - 1] = (u[:, :, 2:m] - 2 * u[:, :, 1:m - 1] + u[:, :, 0:m - 2]) / dxt ** 2
        uxt[:, :, 0] = (u[:, :, 1] + u[:, :, -1] - 2 * u[:, :, 0]) / dxt ** 2
        uxt[:, :, -1] = (u[:, :, 0] + u[:, :, -2] - 2 * u[:, :, -1]) / dxt ** 2
        # uxt[:,:,0] = (2 * u[:,:,0] - 5 * u[:,:,1] + 4 * u[:,:,2] - u[:,:,3]) / dxt ** 2
        # uxt[:,:,n - 1] = (2 * u[:,:,n - 1] - 5 * u[:,:,n - 2] + 4 * u[:,:,n - 3] - u[:,:,n - 4]) / dxt ** 2

    else:
        NotImplementedError()
    # except:
    #     import pdb;pdb.set_trace()

    return uxt


@jit(nopython=True)
def Laplace(u, x):
    x1, x2 = x
    uxt = Diff2_2(u, x1, name=1)
    uxt += Diff2_2(u, x2, name=2)
    return uxt
