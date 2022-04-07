from toolz import curry
import numpy as np
import mymath


@ curry
def cartesian_mask(shape, acc, sample_n=10, centred=False):
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

    for i in range(N):
        start = np.random.randint(0, chunk_size)
        for j in range(Nt):
            mask[i, j, (start + j) % chunk_size::chunk_size] = 1

    mask = np.repeat(mask[..., None], repeats=Ny, axis=-1)

    mask = mask.reshape(shape)

    # for _ in np.sum(mask[0, :, :], axis=-1):
    #     print(_)

    if not centred:
        mask = mymath.ifftshift(mask, axes=(-1, -2))

    return mask
