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
from torch import nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision import datasets
device = "mps" if True else "cuda" if torch.cuda.is_available() else "cpu"

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
"""
for i in range(1, rows * cols + 1):
    ramdom_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[ramdom_idx]
    fig.add_subplot(rows, cols, i)
    plt.title(class_names[label])
    plt.axis(False)
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
"""
#print(train_data)
#print(test_data)
# prepare dataloader
from torch.utils.data import DataLoader
# setup the btach size hyperparameter
BATCH_SIZE = 32

# turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# check out what we've created
# print(f"DataLoeaders: {train_dataloader, test_dataloader}")
# print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE} ...")
# print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE} ...")

# check out whats inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
# print(train_features_batch.shape, train_labels_batch.shape)

# show a saple
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
"""
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
print(f"Image size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")
plt.show()
"""

# Build a baseline model

# create a flatten layer
flatten_model = nn.Flatten()

# get a single sample
x = train_features_batch[0]

# flatten the sample
output = flatten_model(x) # prerform forward pass

# print out what happened
#print(f"Shape before flattering: {x.shape} -> [color_channels, height, width]")
#print(f"Shape before flattering: {output.shape} -> [color_channels, height * width]")

from torch import nn
class FashionMINSTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, hidden_units),
            nn.Linear(hidden_units, output_shape)
        )


    def forward(self, x):
        return self.layer_stack(x)
    
torch.manual_seed(42)

# print((x.shape[1] * x.shape[2]))
# setup model with input parameters
model_0 = FashionMINSTModelV0(
    input_shape = (x.shape[1] * x.shape[2]), # 28 * 28
    hidden_units=10,
    output_shape=len(class_names) # one for every class
).to("cpu")

# setup loss, optimizer and evulation metrics
# loss function for multi-class classification - 'nn.CrossEntropyLoss()'
# optimizer - 'torch.optim.SGD()'
# accuracy as our evaluation metric
from helper_function import accuracy_fn

# setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)

# creating a function to time our experiments
from timeit import default_timer as timer
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """Prints difference between start and end time."""
    total_time = end - start
    print(f"Training time on {device}: {total_time:.3f} seconds")
    return total_time  

# createa training loop and training a model  on batches of data
# loop through epochs
# loop through training batches, perform training steps, calculate the train loss per batch
# loop through testing batches, perform testing steps, calculate the test loss per batch
# print out whats happening
# time it all (for fun)

# tqdm for progress bar
from tqdm.auto import tqdm

# set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# set the number of epochs (we ill keep this small for faster training time)
epochs = 3

# create a training and test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n----__")
    # training
    train_loss = 0
    # add a loop to loop through the training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        # forward pass
        y_pred = model_0(X)

        # calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss

        # optimizer zero grad
        optimizer.zero_grad()
        
        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # print out whats happening
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{train_dataloader.dataset} samples.")

    # divide total train loss by length of train dataloader to get average train loss
    train_loss /= len(train_dataloader)

    # testing
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            # forward pass
            test_pred = model_0(X_test)

            # calculate the loss accumulatively
            test_loss += loss_fn(test_pred, y_test)

            # calculate the accuracy
            test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))

        # calculate the test loss average per batch
        test_loss /= len(test_dataloader)

        # calculate the test acc average per batch
        test_acc /= len(test_dataloader)
    
    # print out the test loss and test accuracy
    print(f"\nTrain loss: {train_loss:.4f}  |  Test loss: {test_loss:.4f})  |  Test_acc: {test_acc:.4f}")

# calculate training time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(
                                start=train_time_start_on_cpu,
                                end=train_time_end_on_cpu,
                                device=str(next(model_0.parameters()).device))

# make predictions and get model_0 results

torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    """returns a dictionary contaning the results of model predicting on data_loader"""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # make predictions
            y_pred = model(X)
            # accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1)) # argmax to get the index of the max value

        # scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# calculate model 0 results on test dataset
model_0_results = eval_model(model_0, test_dataloader, loss_fn, accuracy_fn)

#print(model_0_results)


############# MODEL 1 ############################################

# turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
train_features_batch, train_labels_batch = next(iter(train_dataloader))

test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device=device):
    """returns a dictionary contaning the results of model predicting on data_loader"""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            # make predictions
            y_pred = model(X)
            # accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1)) # argmax to get the index of the max value

        # scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# building a better model with non-linearity
class FashioMINSTModelV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                       out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                       out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)

class_names = train_data.classes

# create an instance of model_1
torch.manual_seed(42)
model_1 = FashioMINSTModelV1(input_shape=train_features_batch.shape[2]*train_features_batch.shape[3],
                             hidden_units=128,
                             output_shape=len(class_names)).to(device)
 
# setup loss, optimizer and evaluation metrics
from helper_function import accuracy_fn
loss_fn = nn.CrossEntropyLoss() # measure how wrong the model is
optmizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1) # update models parameters to reduce the loss
#acc_fn = accuracy_fn # calculate accuracy of model

# functionizing training and evaluating/testing loops
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accutracy_fn,
               device: torch.device = device):
    
    """Performs a training with model trying to learn on data_loader."""
    train_loss, train_acc = 0, 0

    # Put model into training mode
    model.train()

    # add a loop to loop through the training batches
    for batch, (X, y) in enumerate(data_loader):
        # put data on target device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        # calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()
        
        # optimizer step (update the model parameters once per batch)
        optimizer.step()

    # divide total train loss and acc by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f}, Train acc: {train_acc:.2f}%")




def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    """Performs a testing loop step on model goig over data_loader."""
    test_loss, test_acc = 0, 0
    
    # put the model in eval mode
    model.eval()

    # turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # sendo the data to the target device
            X, y = X.to(device), y.to(device)

            # forward pass (output raw logits)
            test_pred = model(X)

            # calculate the loss
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1)) # go from logits -> prediction labels

        # Adjust merics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """Prints difference between start and end time."""
    total_time = end - start
    print(f"Training time on {device}: {total_time:.3f} seconds")
    return total_time  

torch.manual_seed(42)

# measure time
from timeit import default_timer as timer
train_time_start_on_gpu = timer()

# set epochs
epochs = 3

# create a optimization and evaluation loop using train_step() and test_step()
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} \n---------")
    train_step(model=model_1,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optmizer,
               accutracy_fn=accuracy_fn,
               device=device)
    
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
    
train_time_end_on_cgouy = timer()
total_train_time_model_1 = print_train_time(train_time_start_on_gpu,
                                            train_time_end_on_cgouy,
                                            device)

model_1_results = eval_model(model=model_1,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device=device)
                             
#print(model_1_results)
####################################################################################################################################################
####################################################################################################################################################
# Create a convolutional neural network 
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        #print("Output shape of block_1 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX: {x.shape}")
        x = self.block_2(x)
        #print("Output shape of block_2 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX: {x.shape}")
        x = self.classifier(x)
        #print("Output shape of classififier XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX: {x.shape}")
        return x

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1, 
    hidden_units=10, 
    output_shape=len(class_names)).to(device)


torch.manual_seed(42)

# Create sample batch of random numbers with same size as image batch
images = torch.randn(size=(32, 3, 64, 64)) # [batch_size, color_channels, height, width]
test_image = images[0] # get a single image for testing
# print(f"Image batch shape: {images.shape} -> [batch_size, color_channels, height, width]")
print(f"Single image shape: {test_image.shape} -> [color_channels, height, width]") 
# print(f"Single image pixel values:\n{test_image}")

torch.manual_seed(42)

# Create a convolutional layer with same dimensions as TinyVGG 
# (try changing any of the parameters and see what happens)
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0) # also try using "valid" or "same" here 

# Pass the data through the convolutional layer
conv_layer(test_image) # Note: If running PyTorch <1.11.0, this will error because of shape issues (nn.Conv.2d() ex

# Add extra dimension to test image
test_image.unsqueeze(dim=0).shape

# Pass test image with extra dimension through conv_layer
conv_layer(test_image.unsqueeze(dim=0)).shape

torch.manual_seed(42)
# Create a new conv_layer with different values (try setting these to whatever you like)
conv_layer_2 = nn.Conv2d(in_channels=3, # same number of color channels as our input image
                         out_channels=10,
                         kernel_size=(5, 5), # kernel is usually a square so a tuple also works
                         stride=2,
                         padding=0)

# Pass single image through new conv_layer_2 (this calls nn.Conv2d()'s forward() method on the input)
conv_layer_2(test_image.unsqueeze(dim=0)).shape

# Check out the conv_layer_2 internal parameters
print(conv_layer_2.state_dict())

# Get shapes of weight and bias tensors within conv_layer_2
print(f"conv_layer_2 weight shape: \n{conv_layer_2.weight.shape} -> [out_channels=10, in_channels=3, kernel_size=5, kernel_size=5]")
print(f"\nconv_layer_2 bias shape: \n{conv_layer_2.bias.shape} -> [out_channels=10]")

# Print out original image shape without and with unsqueezed dimension
print(f"Test image original shape: {test_image.shape}")
print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}")

# Create a sample nn.MaxPoo2d() layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass data through just the conv_layer
test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")

# Pass data through the max pool layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")

torch.manual_seed(42)
# Create a random tensor with a similiar number of dimensions to our images
random_tensor = torch.randn(size=(1, 1, 2, 2))
print(f"Random tensor:\n{random_tensor}")
print(f"Random tensor shape: {random_tensor.shape}")

# Create a max pool layer
max_pool_layer = nn.MaxPool2d(kernel_size=2) # see what happens when you change the kernel_size value 

# Pass the random tensor through the max pool layer
max_pool_tensor = max_pool_layer(random_tensor)
print(f"\nMax pool tensor:\n{max_pool_tensor} <- this is the maximum value from random_tensor")
print(f"Max pool tensor shape: {max_pool_tensor.shape}")

# plt.imshow(image.squeeze(), cmap="gray")
# plt.show()

rand_image_tensor = torch.randn(size=(1, 28, 28))
print(rand_image_tensor.shape)
model_2(rand_image_tensor.unsqueeze(dim=0).to(device))

# setup a loss function and optimizer for 'model_2'
from helper_function import accuracy_fn

loss_fn = nn.CrossEntropyLoss() # because is a multi-class classification problem
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

# training and testint 'model_2' using our training and test functions
#torch.manual_seed(42)
#torch.mps.manual_seed(42)

# measure time
from timeit import default_timer as timer
train_time_start_model_2 = timer()

# train and test model
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(model=model_2,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accutracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_2,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(train_time_start_model_2,
                                            train_time_end_model_2,
                                            device=device)
# get model 2 results
model_2_results = eval_model(model=model_2,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device=device)
print(model_2_results)

import pandas as pd
compare_results = pd.DataFrame([model_0_results, 
                                model_1_results, 
                                model_2_results])

# add training time to results comparison
compare_results["training_time"] = [total_train_time_model_0,
                                    total_train_time_model_1, 
                                    total_train_time_model_2]
print(compare_results)

# visualize model results
compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
plt.xlabel("Accuracy (%)")
plt.ylabel("Model")
plt.show()

# make and evaluate random predictions with best model
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # prepare the sample *add a batch dimension and pass to target device)
            sample = torch.unsqueeze(sample, dim=0).to(device)  # add a batch dimension
            
            # forward pass (model outputs raw logits)
            pred_logit = model(sample)

            # get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # get pred_prob off the GPU for further calculations
            pred_probs.append(pred_prob.cpu())

        # stack the pred_probs to turn list into a tensor
        return torch.stack(pred_probs)
    
import random
random.seed(42)
test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# make predictions
pred_probs = make_predictions(model_2,data=test_samples)
# view first two prediction probabilities
#print(pred_probs[:2])

# convert prediction probabilities to labels
pred_classes = pred_probs.argmax(dim=1)
#print(pred_classes)

# plot predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    # create subplot
    plt.subplot(nrows, ncols, i+1)

    # plot the targe image
    plt.imshow(sample.squeeze(), cmap="gray")

    # find the prediction (in text form, e.g "sandal")
    pred_label = class_names[pred_classes[i]]
    
    # get the truth label (in text form)
    truth_label = class_names[test_labels[i]]

    # create a title for the plot
    title_text = f"Pred: {pred_label}, Truth: {truth_label}"

    # check for euqality between pred and truth and change color of litle text
    if pred_label == truth_label:
        plt.title(title_text, color="g")
    else:
        plt.title(title_text, color="r")
    plt.axis(False)
plt.show()  

# making a confusion matrix for further prediction evaluation
# make predictions with our trained model on the test dataset
# make a confusion matrix 'torchmetrics.functional.confusion_matrix()'
# plot the confusion matrix using 'mlxtend.plotting.plot_confusion_matrix()'

# Make predictions with trained model
#y_preds = []
model_2.eval()# Primeiro, inicialize y_preds como uma lista vazia antes do loop.
y_preds_list = []

with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions..."):
        # send the data and targets to target device
        X, y = X.to(device), y.to(device)

        # do the forward pass
        y_logit = model_2(X)

        # turn prediction from logits -> prediction probabilities -> prediction labels
        y_preds = torch.softmax(y_logit, dim=1).argmax(dim=1)

        # put prediction on CPU and append to the list
        y_preds_list.append(y_preds.cpu())

# Se você quiser converter sua lista de tensores de volta para um tensor no final:

# concatenate list of prediciton into a tensor
print(f"y_preds: {y_preds}")
y_pred_tensor = torch.cat(y_preds_list)
import os 
# see if required packages are installed and if not, install them
try:
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtendo version should be 0.19.0 or higher"
except:
    os.sys("pip3 install torchmetrics mlxtend")
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")