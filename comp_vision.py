"""
computer vision libraries in pytorch:
'torchvision' is a package in pytorch that has data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz., 'torchvision.datasets' and 'torch.utils.data.DataLoader'
'torchvision.transforms' provides common image transformations such as random crop, rotations, etc.
'torchvision.models' (pretrained computer vision models) has popular architectures such as AlexNet, VGG, ResNet, etc. that can be used to build and train models
'torchvision.utils' has utility functions for image visualization
'torchvision.datasets' has datasets for popular datasets such as Imagenet, CIFAR10, MNIST, etc. that can be easily loaded using 'torchvision.datasets'
'torch.utils.data.DataLoader' is an iterator that provides all these features
"""
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision import datasets

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

#image, label = train_data[0]

class_names = train_data.classes

class_to_idx = train_data.class_to_idx
#class_to_idx = train_data.targets
# check the shape of the image
#print(f"shape of the image: {image.shape} -> [color_channels, height, width]")
#print(f"Image label: {class_names[label]}")

# visualizaing the data
image, label = train_data[0]
# print(f"shape of the image: {image.shape} -> [color_channels, height, width]")
# print(image)
#plt.imshow(image.squeeze() , cmap="gray")
#plt.title(class_names[label])
#plt.axis(False)
#plt.show()

# plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    ramdom_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[ramdom_idx]
    fig.add_subplot(rows, cols, i)
    plt.title(class_names[label])
    plt.axis(False)
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()