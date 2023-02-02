import numpy as np
import cv2

import scipy.ndimage.filters as fi

""" linear algebra """


def dot_vectors(a, b):
    # return np.dot(a, np.array(source_dir)).astype(np.float64)
    return np.sum(np.multiply(a, b), axis=2)


def norm_vectors(m, zero=1.e-9):
    n = np.sqrt(np.sum(np.square(m), axis=2))
    n = np.where(((-1 * zero) < n) & (n < zero), 1, n)
    return n


def normalize_vectors(m):
    norms = norm_vectors(m)
    norms = norms[:, :, np.newaxis]# .repeat(3, axis=2)
    return m / norms


def proj_vectors(u, n):
    # projects vectors onto tangent planes, defined by n
    return dot_vectors(u, n)[:, :, np.newaxis] * normalize_vectors(n)


def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


def partial_derivative(mat, direction):
    assert (direction == 'x' or direction == 'y') \
        , "The derivative direction must be 'x' or 'y'"
    kernel = None
    if direction == 'x':
        # kernel = [[-1.0, 0.0, 1.0]]
        kernel = [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]
    elif direction == 'y':
        # kernel = [[-1.0], [0.0], [1.0]]
        kernel = [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ]
    kernel = np.array(kernel, dtype=np.float32)
    res = cv2.filter2D(mat, -1, kernel)  # / 2.0
    # print('>>', res.shape)
    return res


def normals(s):
    dx = normalize_vectors(partial_derivative(s, 'x'))
    dy = normalize_vectors(partial_derivative(s, 'y'))

    return normalize_vectors(np.cross(dx, dy))
