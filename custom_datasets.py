import torch
from torch import nn

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

import requests
import zipfile
from pathlib import Path

# setup path to a data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# if the image folder doesnt exist, download it and prepare it....
if image_path.is_dir():
    print(f"{image_path}  already exists, skipping download")
else:
    print(f"{image_path} does not exist, downloading...")
    image_path.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/mrdbourke/pytorch-deep-learning/blob/548c91777cf6efc9c4af2010e89bc6b23cfb1a01/data/pizza_steak_sushi.zip"
    
    r = requests.get(url)
    zip_path = data_path / "pizza_steak_sushi.zip"
    zip_path.write_bytes(r.content)
    print("unzipping...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_path)
    print("done")

import os
def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        """Walks through dir_path returning its contents."""
        for dirpath, dirnames, filenames in os.walk(dir_path):
            print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    return dirpath, dirnames, filenames


#walk = walk_through_dir(image_path)
#print(walk)

# visualizing data
import random
from PIL import Image

# set seed
#random.seed(42)

# get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# pick a random image path
random_image_path = random.choice(image_path_list)

# get image class from path name (thi image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# open image
img = Image.open(random_image_path)

# print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
print(f"Image mode: {img.mode}")
#img.show()

import numpy as np
import matplotlib.pyplot as plt

# turn the image into an array
img_as_array = np.array(img)

# plot the image with matplotlib
# plt.figure(figsize=(10, 7))
# plt.imshow(img_as_array)
# plt.title(f"{image_class} image    |    Image shape: {img_as_array.shape}  -> [height, width, color channels] (HWC)")
# plt.axis(False)
#plt.show()

# transforming data
# 1 - turn the target data into tensors (numerical representation of the image class)
# 2 - turn it into a 'touch.utils.data.Dataset' we call tese data Dataset and DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# transforming data with torchvision.transforms
# write a transfm for image
data_transform = transforms.Compose([
    # resize our image to 64x64
    transforms.Resize((64, 64)),
    # flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),
    # turn the image into a tensor with normalized values between 0 and 1
    transforms.ToTensor()
])

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """
    Selects random images from a path of images and loads/transforms
    them then plots the original vs the transformed version.
    """
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, n)
    for image_path in random_image_paths:
        with Image.open(image_path) as img:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(img)
            ax[0].set_title(f"Original\nSize: {img.size}") 
            ax[0].axis(False)

            # transform and ploot target image
            transformed_image = transform(img).permute(1, 2, 0) # we will need to change shape for matplotlib (C, H, W) -> (H, W, C)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
    plt.show()

#plot_transformed_images(image_paths=image_path_list, 
#                        transform=data_transform,
#                         n = 3, seed=None)
# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

# loading image data using 'imageFolder'
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir, 
                                  transform=data_transform, # a transform for the data
                                  target_transform=None)  # transform for the label/target
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

# get class names as list
class_names = train_data.classes

#get class names as dict
class_dict = train_data.class_to_idx

# index on the train_data Dataset to get a single image and label
img, label = train_data[0][0], train_data[0][1]
print(f"Image label: {label}")
print(f"Image tensor: \n {img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print("Label datatype:  {type(label)}")

# rearrange the order dimensions
img_permute = img.permute(1, 2, 0)

# print out different shapes
# print(f"Original shape: {img.shape} -> [color_channels, height, width]")
# print(f"Image permute: {img_permute.shape} -> [height, width, color_channels]")

# plot image
#plt.figure(figsize=(10, 7))
#plt.imshow(img_permute)
#plt.axis("off")
#plt.title(class_names[label], fontsize=14)
# plt.show()

# Turn loaded images into 'DataLoader'
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1,
                              num_workers=0,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=1,
                             num_workers=0,
                             shuffle=False)

img, label = next(iter(train_dataloader))

# batch size will be 1, change batch size if you like
print(f"Image shape: {img.squeeze().shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")

# loading image data with a custom 'Dataset'
# 1 - images from file
# 2 - get class names from the Dataset
# 3 - get classes as dictionary from the Dataset

from typing import Tuple, Dict, List

# instance of torchvision.datasets.ImageFolder()
train_list, train_dict = train_data.classes, train_data.class_to_idx

# creating a helper function to get class names
# get the class names using 'os.scandir()' to traverse a target directory (ideally the directory is in standard image classification format).
# raise an error if the class names arent found (if this happens, there might be something wrong with the directory structure).
# turn the class names into a dict and a list and retur them

# setup path for target directory
target_directory = train_dir
print(f"Target directory: {target_directory}")

# get the class names from the target directory
class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory)) if entry.is_dir()])