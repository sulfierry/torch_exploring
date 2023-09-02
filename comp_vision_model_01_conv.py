import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
BATCH_SIZE = 32


device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# getting a dataset
# setup training data
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(), # transform the image to a tensor
    target_transform=None # transform the target to a tensor
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
