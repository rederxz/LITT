import torch

from model import CRNN_MRI_UniDir
from model_ref import CRNN_MRI_UniDir as CRNN_MRI_UniDir_Ref


# model params

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
# crnn = CRNN_MRI_UniDir()
# crnn_ref = CRNN_MRI_UniDir_Ref()
#
# print(count_parameters(crnn))
# print(count_parameters(crnn_ref))


# dummy input

crnn = CRNN_MRI_UniDir()
dummy = torch.zeros([1, 2, 256, 32, 20])
dummy_output = crnn(dummy, dummy, dummy)

# crnn = CRNN_MRI_UniDir_Ref()
# dummy = torch.zeros([1, 2, 256, 32, 20])
# dummy_output = crnn(dummy, dummy, dummy, useCPU=True)

print(dummy_output.shape)
