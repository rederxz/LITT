import numpy as np

from data import cut_data

dummy = np.ones((32, 3, 256, 256, 5))

output = cut_data(dummy, (256, 32), cut_edge=(0, 32))

print(output.shape)