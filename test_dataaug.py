import numpy as np

from data import cut_data, data_aug

dummy = np.ones((32, 3, 256, 256, 5))

output = cut_data(dummy, (1, 1), cut_edge=(0, 32))
output = data_aug(dummy, (256, 32))

print(output.shape)