""" 
Kinds of transfer learning:
Original mode - Take a pretrained model as it is and apply it to your task without any changes - The original model
reians unchanged - (when to use) helpful if you have the exact same data as the model was trained on.

Feature extraction - Take the underlying patterns (weights) a pretrained model has learned and adjust it outputs
to bem more suited to your prloblem - Use the pretraiend model as a feature extractor (remove the last layer or layers of the pretrained model and replace them with your own custom layers) - (when to use) helpful if you havea small amount of custom adata (simmilar to what the original model was trained on) and want to utilise a pretrained model to get better results on your specific problem.

Fine-tuning - Feature extraction + unfreezing some or all of the layers in the pretrained model and training them on your own custom data - (when to use) helpful if you have a large amount of custom data (similar to what the original model was trained on) and want to tune the pretrained model to your specific problem.
"""

import os
import zipfile
import requests
from pathlib import Path
import torch

# device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# setup data path
data_path = Path('data/')
image_path = data_path / 'pizza_steak_sushi' # images from a subset of classes

# of the path folder desnt existe, download and unzip it
if image_path.is_dir():
    image_path.mkdir(parents=True, exist_ok=True)
    print(f"{image_path} exists, not downloading.")
else:
    print(f"{image_path} desnt exist, downloading now...")  

    # download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        r  = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading zip file...")
        f.write(r.content)
    
    # unzip the downloaded file
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping...")
        zip_ref.extractall(image_path)
    
    # remove .zip file
    os.remove(data_path / "pizza_steak_sushi.zip")

# setup directory path
train_dir = image_path / "train"
test_dir = image_path / "test"

"""
    1 - manually crated transforms - you define what transforms you want your data to go through
    2 - Automatically created transforms - the transforms for you data are defined by the model you'd like to use
    importanto pois: when using a pretrained model, its importante that the data (including your custom data) 
    that you pass through it is transformed in the same way the data the model was originally trained on was transformed.

    All pre-trained models expect input images normalized in the same way, i.e. mini-bacthes of 3-channel RGB images of shape (3 x H x W),where H and W are expected to be at least 224. The images have to loaded in to a range of
    [0, 1] and then normalized using mean and std. You can use the following transform to normalize:
"""


"""MANUALLY"""

from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.546, 0.406],
                                std=[0.229, 0.224, 0.225])

manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # height x width
    transforms.ToTensor(), # get images into range [0, 1]
    normalize]) # make sure image have the same distribuition as the ones the model was trained on


import data_setup
from data_setup import create_dataloaders

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                             test_dir=test_dir,
                                                                             transform=manual_transforms,
                                                                             batch_size=32)
#print(train_dataloader, test_dataloader, classname)

"""AUTOMATICALLY"""
# creating a transform for 'torchvision.models' (auto creation)
import torchvision

# get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

# get the transforms used to create our pretrained wieghts
auto_transforms = weights.transforms()
print(auto_transforms)
# create dataloaders using automatic transforms
train_dataloader, test_dataloader, calss_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=auto_transforms,
    batch_size=32 
)

print(train_dataloader, test_dataloader, calss_names)

"""
    getting a pretrained model:
    1 - PyTorch domain libraries
    2 - Libraries like timm (torch image models)
    3 - HuggingFace's Transformers library
    4 - Scientific papers

    which pretrained model shoud you use? Experiment!
    The whole idea of transfer learning: take an already well-performing model from a
    problem space similar to you own and then adapt it to your own problem space.

    three things to consider:
    1 - Speed: how fast does it run?
    2 - Size: how big is the model?
    3 - Performance: how well does it perform?
"""

# setting up a pretrained model using a instace of EffNetB0
#model = torchvision.models.efficientnet_b0(pretrained=True) # pretrained on ImageNet

# new method of creartgin a pretrained model
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT best avaliable weights
model = torchvision.models.efficientnet_b0(weights=weights).to(device)
#print(model.classifier)

# getting a summary of our model with 'torchinfo.summary()'
from torchinfo import summary

summary(model=model, 
        input_size=(1, 3, 224, 224), # [btach_size, color_channels, height, width]
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

# freezing the base model and changing the output layer to suit our needs
for param in model.features.parameters():
    param.requires_grad = False # freeze the base model
    


#  update the classifier head of our model to suit our ploblem
import torch
from torch import nn

torch.manual_seed(42)
# torch.mps.manual_seed(42)
model.classifier = nn.Sequential(

    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, # feature vector coming in
              out_features=len(class_names))).to(device)


summary(model=model, 
        input_size=(1, 3, 224, 224), # [btach_size, color_channels, height, width]
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

print(model.classifier)

# train model
# define lloss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# import train function
import engine 

torch.manual_seed(42)
# torch.mps.manual_seed(42)

# start the timer   
from timeit import default_timer as timer
start_timer = timer()

results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=5,
                       device=device)

# end the timer and print out how long it took
end_timer = timer()
print(f"Time taken to train: {end_timer - start_timer}")

# evaluate model by plotting loss curves
from helper_functions import plot_loss_curves
#plot_loss_curves(results)

""""
make predictions on images from the test set, needs same:
    shapes
    datattype
    device
    transform

    to do all of this automatically, lets create a function called 'pred_and_plot_image():
    1 - take in a trained mode,, a list of calss names, a filepath to targe image, an image size, a transform and a target device.
    2 - Open the image with 'PILL.Image.open()' and convert it to RGB (if it's not already)
    3 - Create a tranform if one desnt exist
    4 - Make sure the model in on the target device
    5 - Turn the model to 'model.eval()' mode to make sure its ready for inference
    6 - Transform the target image and make sure its dimensionality is suited for the model (this mainly relates to batch size)
    7 - Make a prediction on the image by passing to the model
    8 - Convert the model output logits to prediction probabilities using 'torch.softmax()'
    9 - COnvert models prediction probabilities to prediction labels using 'torch.argmax()'
    10 - Plot he image with 'matplotlib' and set the title to the predcition label from step 9 and prediction probability from step 8

"""

from torchvision import transforms
import matplotlib.pyplot as plt
from typing import Tuple, List
from PIL import Image

# 1 - take in a trained model
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):
    # 2 - open the image with PIL
    img = Image.open(image_path)

    # 3 - create a transform if one desnt exist
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.546, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    ### predict on image ###
    # 4 - make sure the model is on the target device
    model.to(device)

    # 5 - turn the model to eval mode
    model.eval()
    with torch.inference_mode():
        # 6 - transform the iamge and add an extra batch dimension
        transformed_image = image_transform(img).unsqueeze(dim=0) # [batch_size, color_channels, height, width]

        # 7 - make a prediction on the image by passing it to the model
        target_image_pred = model(transformed_image.to(device))
    
    # 8 - convert the model output logits to prediction probabilities using 'torch.softmax()'
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9 - convert the models prediction probabilities to prediction labels using 'torch.argmax()'
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10 - plot the image with matplotlib and set the title to the prediction label from step 9 and prediction probability from step 8
    plt.figure()
    plt.imshow(img)
    plt.title(f"Prediction: {class_names[target_image_pred_label]} | Probabilite: {target_image_pred_probs.max()}")
    plt.axis(False)
    plt.show()

# get a random list of image paths from the test set
import random
num_images_to_plot = 3
test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))
test_image_path_sample = random.sample(population=test_image_path_list, 
                                       k=num_images_to_plot)
# make predicitions on and plot the images
for image_path in test_image_path_sample:
    pred_and_plot_image(model=model,
                        image_path=image_path,
                        class_names=class_names,
                        image_size=(224, 224),
                        transform=None,
                        device=device)

custom_image_path = "data/04-pizza-dad.jpeg"

# making prediction on a custom image
pred_and_plot_image(model=model,
                    image_path=custom_image_path,
                    class_names=class_names)