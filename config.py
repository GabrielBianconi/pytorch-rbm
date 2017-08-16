import torch

CUDA = True
CUDA_DEVICE = 0

if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

DATA_FOLDER = 'data/mnist'
