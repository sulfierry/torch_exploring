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

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder name sin a target directory"""
    # get the class names by scanning the target directory
    classes = sorted([entry.name for entry in os.scandir(directory) if entry.is_dir()])

    # raise an error if class names could not be found
    if not classes:
        raise FileNotFoundError(f"Could not find any class folders in {directory}.")  

    # create a dicttionary of index labels 
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    return classes, class_to_idx

find_classes(target_directory)
"""
1 - subclass 'torch.utils.data.Dataset'
2 - unit our subclass with a target directory
3 - create several atributes:
    paths - paths of our images
    transform - the transform wed like to use
    classes - a list of the target classes
    class_to_idx - a dict of the target classes mapped of our dataset
4 - Create a function to load_images(), thhis function will open an image
5 - Overwrite __len__ and __getitem__ to make our dataset iterable
"""
# write a custom dataset class
from torch.utils.data import Dataset
import pathlib
# 1 - subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    # 2 - initialize our custom dataset
    def __init__(self, targ_dir: str, transform=None):
        # 3 - create class atributes
        # get all of the image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        # setup transform
        self.transform = transform
        # Create classes and class_to_idx
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4 - create a function to load images
    def load_image(self, index: int) -> Image.Image:
        """Opens an image via a path and returns it."""
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5 - overwrite __len__()
    def __len__(self) -> int:
        """return the total number of samples."""
        return len(self.paths)
    
    # 6 - overwrite __getitem__() method to return a particular sample
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """"Returns one sample of data, data and label (X, y)."""
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # expects path in format: 'data_folder/class_name/image.jpg'
        class_idx = self.class_to_idx[class_name]
    
        # transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return untransformed data, label (X, y)

from torchvision import transforms

# create a transform
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# test out ImageFolderCustom
train_data_custom = ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)

# check for equality between original ImageFolder Dataset and ImageFolderCustom Dataset
#print(train_data_custom.classes==train_data.classes)
#print(test_data_custom.classes==test_data.classes)
"""
create a function to display random images
1 - take in a 'Dataset' and a number of other parameter such as class names and how many images to visualize
2 - to rpevent the display getting out ofhand, lets cap the number of images to see at 10
3 - set the random seed for reproducibility
4 - get a list of random sample indexes from the target dataset
5 - setup a matlpotlib plot.
6 - Loop through the random sample images and plot them with matplotlib
7 - Make sure the dimensions of our images line up with matplotlib (HWC)
"""

# 1 - create a function to take in a dataset
def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    
    # 2 - adjust display if n is to high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display, purposes, n, shouldnt be larger than 10, setting to 10 and removing shape display.")
    
    # 3 - set the seed
    if seed:
        random.seed(seed)
    
    # 4 - get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5 - setup plot
    plt.figure(figsize=(16, 8))

    # 6 - loop through the samples indexes and plot them with matplotlib
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7 - adjust tensor dimensions for plotting
        targ_image_adjust = targ_image.permute(1, 2, 0) # [color_channels, height, width] -> [height, width, color_channels]

        # plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")

        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape:
                title += f"\nShape: {targ_image_adjust.shape}"
        plt.title(title)
    plt.show()

# display random images from the ImageFolderCustom Dataset
# display_random_images(dataset=train_data,
#                       classes=class_names,
#                       n=5,
#                       display_shape=True,
#                       seed=None)
# doisplay random images from the ImageFolderCustom Dataset
display_random_images(train_data_custom,
                      n=20,
                      classes=class_names,
                      seed=None)

