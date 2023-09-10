# Vision Transformer (ViT) https://doi.org/10.48550/arXiv.2010.11929

# For this notebook to run with updated APIs, we need torch 1.12+ and torchvision 0.13+
from data_setup import create_dataloaders
from helper_functions import download_data
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torch


# setup device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# download data
image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")
train_diir = image_path / "train"
test_dir = image_path / "test"

# create datasets and dataloaders
train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_diir,
                                                                    test_dir=test_dir,
                                                                    transform=torchvision.transforms.ToTensor(),
                                                                    batch_size=32,
                                                                    num_workers=4)
# check page 13, table 3 from paper
IMG_SIZE = 224 # 224 x 224 px
NUM_CLASSES = len(class_names) # 3 classes

# creating image patches and turning them into patch embeddings