import torch

from model import CRNN
from model_ref import CRNN_MRI_UniDir as CRNN_MRI_UniDir_Ref


# model params

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
# crnn = CRNN(uni_direction=True)
# crnn_ref = CRNN_MRI_UniDir_Ref()
#
# print(count_parameters(crnn))
# print(count_parameters(crnn_ref))


# dummy input

dummy = torch.zeros([1, 2, 256, 32, 20])

crnn = CRNN()
dummy_output = crnn(dummy, dummy, dummy)

crnn2 = CRNN_MRI_UniDir_Ref()
dummy_output2 = crnn2(dummy, dummy, dummy, useCPU=True)

print(dummy_output.shape)
print(dummy_output2.shape)
