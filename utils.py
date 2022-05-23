import math
import time

import numpy as np
from numpy.fft import fftshift, fft, ifftshift, ifft, fft2, ifft2
from numpy.lib.stride_tricks import as_strided
from scipy import signal
from toolz import curry


def r2c(x, axis=1):
    """Convert pseudo-complex data (2 real channels) to complex data

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    if axis < 0: axis = x.ndim + axis
    ctype = np.complex64 if x.dtype == np.float32 else np.complex128

    if axis < len(shape):
        newshape = tuple([i for i in range(0, axis)]) \
                   + tuple([i for i in range(axis+1, x.ndim)]) + (axis,)

        x = x.transpose(newshape)

    x = np.ascontiguousarray(x).view(dtype=ctype)
    return x.reshape(x.shape[:-1])


def c2r(x, axis=1):
    """Convert complex data to pseudo-complex data (2 real channels)

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    dtype = np.float32 if x.dtype == np.complex64 else np.float64

    x = np.ascontiguousarray(x).view(dtype=dtype).reshape(shape + (2,))

    n = x.ndim
    if axis < 0: axis = n + axis
    if axis < n:
        newshape = tuple([i for i in range(0, axis)]) + (n-1,) \
                   + tuple([i for i in range(axis, n-1)])
        x = x.transpose(newshape)

    return x


def mask_r2c(m):
    return m[0] if m.ndim == 3 else m[:, 0]


def to_tensor_format(x, mask=False):
    """
    Assumes data is of shape (n[, nt], nx, ny).
    Reshapes to (n, n_channels, nx, ny[, nt])
    Note: Depth must be the last axis, the dimensions will be reordered
    """
    if x.ndim == 4:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 2, 3, 1))

    if mask:  # Hacky solution
        x = x*(1+1j)

    x = c2r(x)

    return x


def from_tensor_format(x, mask=False):
    """
    Assumes data is of shape (n, 2, nx, ny[, nt]).
    Reshapes to (n, [nt, ]nx, ny)
    """
    if x.ndim == 5:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 1, 4, 2, 3))

    if mask:
        x = mask_r2c(x)
    else:
        x = r2c(x)

    return x


def fftc(x, axis=-1, norm='ortho'):
    ''' expect x as m*n matrix '''
    return fftshift(fft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


def ifftc(x, axis=-1, norm='ortho'):
    ''' expect x as m*n matrix '''
    return fftshift(ifft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


def fft2c(x):
    '''
    Centered fft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    # axes = (len(x.shape)-2, len(x.shape)-1)  # get last 2 axes
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(fft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res


def ifft2c(x):
    '''
    Centered ifft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(ifft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res


def fourier_matrix(rows, cols):
    '''
    parameters:
    rows: number or rows
    cols: number of columns

    return unitary (rows x cols) fourier matrix
    '''
    # from scipy.linalg import dft
    # return dft(rows,scale='sqrtn')

    col_range = np.arange(cols)
    row_range = np.arange(rows)
    scale = 1 / np.sqrt(cols)

    coeffs = np.outer(row_range, col_range)
    fourier_matrix = np.exp(coeffs * (-2. * np.pi * 1j / cols)) * scale

    return fourier_matrix


def inverse_fourier_matrix(rows, cols):
    return np.array(np.matrix(fourier_matrix(rows, cols)).getH())


def flip(m, axis):
    """
    ==== > Only in numpy 1.12 < =====

    Reverse the order of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.
    .. versionadded:: 1.12.0
    Parameters
    ----------
    m : array_like
        Input array.
    axis : integer
        Axis in array, which entries are reversed.
    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.
    See Also
    --------
    flipud : Flip an array vertically (axis=0).
    fliplr : Flip an array horizontally (axis=1).
    Notes
    -----
    flip(m, 0) is equivalent to flipud(m).
    flip(m, 1) is equivalent to fliplr(m).
    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.
    Examples
    --------
    >>> A = np.arange(8).reshape((2,2,2))
    >>> A
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> flip(A, 0)
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    >>> flip(A, 1)
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    >>> A = np.random.randn(3,4,5)
    >>> np.all(flip(A,2) == A[:,:,::-1,...])
    True
    """
    if not hasattr(m, 'ndim'):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]


def rot90_nd(x, axes=(-2, -1), k=1):
    """Rotates selected axes"""
    def flipud(x):
        return flip(x, axes[0])

    def fliplr(x):
        return flip(x, axes[1])

    x = np.asanyarray(x)
    if x.ndim < 2:
        raise ValueError("Input must >= 2-d.")
    k = k % 4
    if k == 0:
        return x
    elif k == 1:
        return fliplr(x).swapaxes(*axes)
    elif k == 2:
        return fliplr(flipud(x))
    else:
        # k == 3
        return fliplr(x.swapaxes(*axes))


@ curry
def kt_blast_cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nt, nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    sample_n: preserve how many center lines (sample_n // 2 each side)
    """
    N, Nt, Nx, Ny = int(np.prod(shape[:-3])), shape[-3], shape[-2], shape[-1]

    chunk_size = int(acc)  # one line sampled per chunk

    mask = np.zeros((N, Nt, Nx))

    if sample_n:
        mask[..., Nx // 2 - sample_n // 2: Nx // 2 + sample_n // 2] = 1

    step = int((acc + 1) // 2) if acc % 2 == 1 else math.floor(acc / 7) * 2 + 1

    for i in range(N):
        start = np.random.randint(0, chunk_size)
        for j in range(Nt):
            mask[i, j, (start + j * step) % chunk_size::chunk_size] = 1

    mask = np.repeat(mask[..., None], repeats=Ny, axis=-1)

    mask = mask.reshape(shape)

    # for _ in np.sum(mask[0, :, :], axis=-1):
    #     print(_)

    if not centred:
        mask = ifftshift(mask, axes=(-1, -2))

    return mask


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


@ curry
def cs_cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    sample_n: preserve how many center lines (sample_n // 2 each side)

    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = ifftshift(mask, axes=(-1, -2))

    return mask


@ curry
def low_resolution_cartesian_mask(shape, acc, centred=False):
    """
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]

    sample_n = int(Nx / acc)

    mask = np.zeros((N, Nx))
    mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    mask = np.repeat(mask[..., None], repeats=Ny, axis=-1)

    mask = mask.reshape(shape)

    if not centred:
        mask = ifftshift(mask, axes=(-1, -2))

    return mask


def undersample(x, mask, centred=False, norm='ortho', noise=0):
    '''
    Undersample x. FFT2 will be applied to the last 2 axis
    Parameters
    ----------
    x: array_like
        data
    mask: array_like
        undersampling mask in fourier domain

    norm: 'ortho' or None
        if 'ortho', performs unitary transform, otherwise normal dft

    noise_power: float
        simulates acquisition noise, complex AWG noise.
        must be percentage of the peak signal

    Returns
    -------
    xu: array_like
        undersampled image in image domain. Note that it is complex valued

    x_fu: array_like
        undersampled data in k-space

    '''
    assert x.shape == mask.shape
    # zero mean complex Gaussian noise
    noise_power = noise
    nz = np.sqrt(.5)*(np.random.normal(0, 1, x.shape) + 1j * np.random.normal(0, 1, x.shape))
    nz = nz * np.sqrt(noise_power)

    if norm == 'ortho':
        # multiplicative factor
        nz = nz * np.sqrt(np.prod(mask.shape[-2:]))
    else:
        nz = nz * np.prod(mask.shape[-2:])

    if centred:
        x_f = fft2c(x, norm=norm)
        x_fu = mask * (x_f + nz)
        x_u = ifft2c(x_fu, norm=norm)
        return x_u, x_fu
    else:
        x_f = fft2(x, norm=norm)
        x_fu = mask * (x_f + nz)
        x_u = ifft2(x_fu, norm=norm)
        return x_u, x_fu


def mag_min_max_normalize(x):
    """
    Args:
        x: [t, x, y]

    Returns:
        magnitude min_max_normalized x
    """
    for i, frame in enumerate(x):
        angle, mag = np.angle(frame), np.abs(frame)
        mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag))
        x[i] = mag * np.exp(1j * angle)

    return x


def mag_min_max_normalize_clip(x, clip_percentage=5):
    """
    Args:
        x: [t, x, y]

    Returns:
        magnitude min_max_normalized x
    """
    for i, frame in enumerate(x):
        angle, mag = np.angle(frame), np.abs(frame)
        mag_max = np.percentile(mag, 100 - clip_percentage)
        mag_min = np.percentile(mag, clip_percentage)
        mag[mag > mag_max] = mag_max
        mag[mag < mag_min] = mag_min
        mag = (mag - mag_min) / (mag_max - mag_min)
        x[i] = mag * np.exp(1j * angle)

    return x


def coil_combine(multi_coil_img, filt_cor=False):
    """
    Args:
        multi_coil_img: [nt, nx, ny, ncoil]
        filt_cor: whether filter correlation matrix

    Returns:
        combined_img: [nt, nx, ny]
    """

    nt, nx, ny, ncoil = multi_coil_img.shape

    print(f'Calculating correlation matrix ...')
    tik = time.time()
    # correlation matrix [nx, ny, ncoil, ncoil], only calculate on the last frame to save time
    cor = np.matmul(multi_coil_img[-1, ..., None], np.conj(multi_coil_img)[-1, ..., None, :])
    if filt_cor:
        for i in range(ncoil):
            for j in range(ncoil):
                cor[..., i, j] = signal.convolve2d(cor[..., i, j], np.ones([3, 3]), mode='same')
    tok = time.time()
    print(f'Finish in {tok - tik:.4f}s.')

    # compute filter
    print(f'SVD ...')
    tik = time.time()
    u, s, vh = np.linalg.svd(cor)
    filt = np.conj(u[..., 0][..., None, :])
    tok = time.time()
    print(f'Finish in {tok - tik:.4f}s.')

    # apply filter
    combined_imgs = list()
    for i in range(nt):
        print(f'Combining frame {i + 1}/{nt} ...')
        tik = time.time()
        combined_img = np.matmul(filt, multi_coil_img[i, ..., None])[..., 0, 0]
        tok = time.time()
        print(f'Finish in {tok - tik:.4f}s.')
        combined_imgs.append(combined_img)

    combined_imgs = np.stack(combined_imgs, axis=0)

    return combined_imgs
