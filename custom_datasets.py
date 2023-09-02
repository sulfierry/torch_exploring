import torch
import random
import zipfile
import pathlib
import requests
import torchvision
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from torchinfo import summary
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from torchvision.transforms import Resize
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset



device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"



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

# # print metadata
# print(f"Random image path: {random_image_path}")
# print(f"Image class: {image_class}")
# print(f"Image height: {img.height}")
# print(f"Image width: {img.width}")
# print(f"Image mode: {img.mode}")
#img.show()



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
# print(f"Image label: {label}")
# print(f"Image tensor: \n {img}")
# print(f"Image shape: {img.shape}")
# print(f"Image datatype: {img.dtype}")
# print("Label datatype:  {type(label)}")

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
# print(f"Image shape: {img.squeeze().shape} -> [batch_size, color_channels, height, width]")
# print(f"Label shape: {label.shape}")

# loading image data with a custom 'Dataset'
# 1 - images from file
# 2 - get class names from the Dataset
# 3 - get classes as dictionary from the Dataset

# instance of torchvision.datasets.ImageFolder()
train_list, train_dict = train_data.classes, train_data.class_to_idx

# creating a helper function to get class names
# get the class names using 'os.scandir()' to traverse a target directory (ideally the directory is in standard image classification format).
# raise an error if the class names arent found (if this happens, there might be something wrong with the directory structure).
# turn the class names into a dict and a list and retur them

# setup path for target directory
target_directory = train_dir
# print(f"Target directory: {target_directory}")

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
        plt.title(title, fontsize=5)
    plt.show()

# display random images from the ImageFolderCustom Dataset
# display_random_images(dataset=train_data,
#                       classes=class_names,
#                       n=5,
#                       display_shape=True,
#                       seed=None)

# display random images from the ImageFolderCustom Dataset
# display_random_images(train_data_custom,
#                       n=10,
#                       classes=class_names,
#                       seed=None)

# Turn custom loaded imates into 'DataLoader'

BATCH_SIZE = 64
NUM_WORKERS = 0
train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        shuffle=True)

test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        shuffle=False)

# get image and label from custom dataloader
img_custom, label_custom = next(iter(train_dataloader_custom))

# data argumentation is the process of artificially adding diversity to your training data
# lets look a trivialaugment

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), # commom size in image classifciation
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # TrivialAugmentWide is a custom transform from the timm library
    transforms.ToTensor()
    ])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

# get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# # plot random images
# plot_transformed_images(image_paths=image_path_list,
#                         transform=train_transform,
#                         n=3,
#                         seed=None)

# model_0: TinyVGG without data augmentation
# creating transforms and loading data for model_0
simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# load and transform data
train_data_simple = datasets.ImageFolder(root=train_dir,
                                         transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir,
                                        transform=simple_transform)
# turn the datasets into DataLoaders

# setup batch size and number of workers
BATCH_SIZE = 32
NUM_WORKERS = 0 # os.cpu_count()

# create dataloaders
train_dataloader_simple = DataLoader(dataset=train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=NUM_WORKERS)
# create TinyVGG model class
class TinyVGG(nn.Module):
    """Model architecture copying TinyVGG from CNN explainer."""
    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13,
                      out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x # self.classifier(self.conv_block_2(self.conv_block_1(x))) # benefits from operator fusion


torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3, # number of color channels in image data
                  hidden_units=10, # number of hidden units in a conv layer
                  output_shape=len(class_names)).to(device) # number of classes in the data
#print(model_0)

# try a forward pass on a single image (to test the model)
# get a single image batch
image_batch, label_batch = next(iter(train_dataloader_simple))

# try a forward pass
#print(model_0(image_batch.to(device)))



summary(model_0, input_size=(1, 3, 64, 64))

# create train  and test loops functions

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # put the model into training mode
    model.train()

    # setup trian loss and train accuracy values
    train_loss, train_acc = 0, 0

    # loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # send data to the target device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X) # output model logits

        # calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward() 

        # optimizer stop
        optimizer.step()

        # calculate accuracy metric
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y)
    
    # adjust metrics to get average loss and accuracy per betch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device=device):
    # put model in eval mode
    model.eval()

    # setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # turn on inference mode
    with torch.inference_mode():
        # loop through Dataloader batches
        for batch, (X, y) in enumerate(dataloader):
            # send data to the target device
            X, y = X.to(device), y.to(device)

            # forward pass
            test_pred_logits = model(X)

            # calculate the loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # calculate the accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(y))
    
    # adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# creating a 'train()' function to combine train_step() and test_step()

# create a train fcuntion that takes in various models paremeters 
def train(model: torch.nn.Module,
          train_dataloader,
          test_dataloader,
          optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device=device):
    
    # create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
                "test_loss": [],
                "test_acc": []}
    
    # loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        # printou whats happening
        print(f"Epoch: {epoch+1} | Train loss: {train_loss:.5f} | Train acc: {train_acc:.5f} | Test loss: {test_loss:.5f} | Test acc: {test_acc:.5f}")

        # update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

# plot the loss cruves of model_0
# get the model_0_results keys

def plot_loss_curves(results: Dict[str, List[float]]) -> None:
    """Plots training curves of a results dictionary"""
    # get the loss values of the results dictionary (training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # get the accuracy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # figure out how many epochs there were
    epochs = range(len(results["train_loss"]))

    # setup a plot
    plt.figure(figsize=(15, 7))

    # plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def plot_train_test_duplicate(model_0_df, model_1_df):
    # plot train loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
    plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.legend()
    #plt.show()

    # plot test loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
    plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
    plt.title("Test Loss")
    plt.xlabel("Epochs")
    plt.legend()
    #plt.show()

    # accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
    plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
    plt.title("Train Acc")
    plt.xlabel("Epochs")
    plt.legend()

    # accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
    plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
    plt.title("Test Acc")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
#########################################################################################################
# set random seeds
torch.manual_seed(42)
torch.mps.manual_seed(42)

# set number of epochs 
NUM_EPOCHS = 5

# recreate an instance of TinyVGG
model_0 = TinyVGG(input_shape=3, # number of color channels
                  hidden_units=10, # number of hidden units in a conv layer
                  output_shape=len(class_names)).to(device) # number of classes in the data

# setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)

# start the timer 
from timeit import default_timer as timer
start_time = timer()

# train model_0
model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)
# end the timer
end_timer = timer()
print(f"Time taken to train model_0: {end_timer - start_time:.3f} seconds")


#plot_loss_curves(model_0_results)

"""
Dealing with overfitting
get more data:            giver a model more of a chance to learn patterns between samples
data argumentation:       increase the diversity of your training dataset without collecting more data. Increased diversity forces a model to learn more generalisation patterns.
better data:              not all data samples are created euqally. Removing poor samples from or adding better samples to you dataset can improve your model performance.
use transfer learning:    use the patterns learned by a model trained on a large dataset to help your model learn patterns on your own dataset.
simplify your model:      the less parameters a model has, the less patterns it can learn. If your model is learning too many patterns, it might be overfitting.
user learning rate decay: gradually reduce the learning rate as the model trains. This allows the model to make large updates in the beginning of training (helping it find patterns) and smaller updates later on (helping it fine-tune the patterns).
use early stopping:       training for too long can cause a model to learn patterns on the training data which don't generalise to the test data. Early stopping stops a model from training once it stops improving on the test set.
"""

# model_1 TinyVGG with data augmentation
# create trining trnasforms with data TrivialAugmentWide
train_transform_trivial = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
    ])

test_transform_simple = transforms.Compose([ # we dont do data argumentation on the test data
    transforms.Resize((64, 64)),
    transforms.ToTensor()
    ])

# create train and test 'Datasets' and 'DataLoaders' with data argumentation
# turn image folder into Datasets
train_data_augmented = datasets.ImageFolder(root=train_dir,
                                            transform=train_transform_trivial)
test_data_augmented = datasets.ImageFolder(root=test_dir,
                                           transform=test_transform_simple)

# turn our datasets into dataloaders
BATCH_SIZE = 32
NUM_WORKERS = 0

torch.manual_seed(42)
train_dataloader_augmented = DataLoader(dataset=train_data_augmented,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(dataset=test_data_augmented,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=NUM_WORKERS)
# construct and train model_1
model_1 = TinyVGG(input_shape=3,
                    hidden_units=10,
                    output_shape=len(train_data_augmented.classes)).to(device)

# setup loss function and optimizer
# set random seeds
torch.manual_seed(42)
torch.mps.manual_seed(42)

NUM_EPOCHS = 5 # number of epochs 
loss_fn = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.001) # optimizer
start_time = timer() 

# train model_1
model_1_results = train(model=model_1,
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS,
                        device=device)
end_timer = timer()
#plot_loss_curves(model_1_results)
print(f"Time taken to train model_1: {end_timer - start_time:.3f} seconds.")

# compare model results
model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)

# setup a plot
plt.figure(figsize=(15, 10))

# get number of epochs
epochs = range(len(model_0_df))
# plot_train_test_duplicate(model_0_df, model_1_df)

# setup custom image path
custom_image_path = data_path / "04-pizza-dad.jpeg"

# load in a custom image with pytorch
# make sure taht the iamge is in the same formata as the data of the model
"""
    in tensor form with datatype torch.float32
    of shape 64 x 64 x 64
    on the right device
"""
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
custom_image = custom_image / 255 # normalize the image

print(f"Custom image shape: {custom_image.shape}")
print(f"Custom image datatype: {custom_image.dtype}")
print(f"Custom image tensor: \n {custom_image}")
# plt.imshow(custom_image_uint8.permute(1, 2, 0))
# plt.axis(False)
# plt.show()
# making a prediction on a custom image with a trained pytorch model
# try to make a prediction on a image in uint8 format
#model_1.eval()
#with torch.inference_mode():
#    model_1(custom_image.to(device))

# create transform pipeline to resize image


# transform target image
# Create transform pipleine to resize image
custom_image_transform = Resize(size=(64, 64), antialias=True)

# Transform target image
custom_image_transformed = custom_image_transform(custom_image)

# Print out original shape and new shape
print(f"Original shape: {custom_image.shape}")
print(f"New shape: {custom_image_transformed.shape}")

"""Shape error bellow:"""
# (mpsFileLoc): /AppleInternal/Library/BuildRoots/d9889869-120b-11ee-b796-7a03568b17ac/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShadersGraph/
# mpsgraph/MetalPerformanceShadersGraph/Core/Files/MPSGraphUtilities.mm:39:0: error: 'mps.matmul' op contracting dimensions differ 169 & 1690

model_1.eval()
with torch.inference_mode():
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(0).to(device))
"""
    To make a prediction on a custom image we had to:
        load the image and tur it into a tensor
        make sure the image was the same datatype as the model (torch.float32)
        make sure the was the same shape as the data the model was trained on (3, 64, 64) with a batch size (1, 3, 64, 64)
        make sure the image was on the same device as our model
"""

# convert logits -> prediction probabilities
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)

# convert prediction probabilities -> prediction labels
custom_image_pred_labels = torch.argmax(custom_image_pred_probs, dim=1)
print(class_names[custom_image_pred_labels])

# putting custom image prediction together: building a function
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device=device):
    """Makes a prediction on a target image with a trained model and plotes the imate and prediction."""
    # load target image
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.

    # transform if necessary
    if transform:
        target_image = transform(target_image)
    
    # make sure the model is on the target device
    model.to(device)

    # turn on eval/inference mode and make prediction
    model.eval()
    with torch.inference_mode():
        # add an extra dimension to the image (his is the batch dimension, the model will predict on batches of 1x image)
        target_image = target_image.unsqueeze(0)
        
        # make predicitoon on the image with an extra dimension
        target_image_pred = model(target_image.to(device))
    
    # convert logits -> prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # convert prediction probabilities -> prediction labels
    target_image_pred_labels = torch.argmax(target_image_pred_probs, dim=1)

    # plot the image alongside the predicction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # remove batch dimension and rearrange shape to be HWC
    if class_names:
        title = f"Pred: {class_names[target_image_pred_labels.cpu()]}  |  Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_labels}  |  Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()

# pred on our custom image
pred_and_plot_image(model=model_1,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)
