# Make classification data and get it ready
from sklearn.datasets import make_circles

# make samples
n_samples = 1000
# create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

#print(len(X), len(y))
#print(f"First 5 samples of X:\n {X[:5]}")
#print(f"First 5 samples of y:\n {y[:5]}")

# make dataframe of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})

#print(circles.head(10))
#print(circles.label.value_counts()) 
# visualize
import matplotlib.pyplot as plt
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
#plt.show()

# view first example of features and labels
X_sample = X[0]
y_sample = y[0]

#print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
#print(f"Values for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

# Turn data into tensors
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
#print(X[:5], y[:5])
torch.manual_seed(42)
# split data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 0.2 -> 20% will be test & 80% will be training
                                                    random_state=42) 
#print(len(X_train), len(X_test), len(y_train), len(y_test))

# build a model to classify blue and red dots, we wanto to:
# setup device agnostic code so it can run on CPU or GPU
# construct a model (by subclassing 'nn.Module')
# define a loss function and optimizer
# create a training and test loop
import torch.nn as nn

# make device agnostic code
device = "mps" if True else "cuda" if torch.cuda.is_available() else "cpu"

# subcalss nn.Module 
# create nn.Linar() - label that are capable of handling the shapes of our data
# Defines a forward() - outlines the forward pass of the model
# Instatiate an instance of our model class and send it to the target device

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features and upscale to 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features and downscale to 1 feature

    def float(self, x):
        return self.layer_2(self.layer_1(x)) # x _> layer_1 -> layer_2 -> out
    
# instantiate an instance of our model class and send it to the target device
model_0 = CircleModelV0().to(device)
#print(model_0)

# replicate the model above using nn.Sequential()
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

#print(model_0)

# make predicitons
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))

#print(f"Length of predicition: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
#print(f"Length of samples: {len(X_test)}, Shape: {X_test.shape}")
#print(f"\nFirst 10 predictions: \n{untrained_preds[:10]}")
#print(f"\nFirst of labels: \n{y_test[:10]}")

# Setup loss function and optimizer
# for regession MAE or MSE (Mean Absolute Error or Mean Squared Error)
# for classification binary cross entropy or categorical cross enrtropy (cross entropy)
# for optimizer SGD, Adam, RMSProp
# for loss function we use 'torch.nn.BECWithLogitsLoss()', 'binary_cross_entropy_with_logits()'
 
 
# setup the loss function
# requires inputs to have gone through the sigmoid activation function prior to input to BCEWithLogitsLoss
loss_fn = nn.BCEWithLogitsLoss() # sigmoid activation function + binary cross entropy loss

optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.01)

#print(optimizer.state_dict())
# calculate accuracy - out of 100 examples, what percentage get right?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_true)) * 100

# train model
# forward pass
# calculate the loss
# optmizer zero grad
# loss backward (backpropagation)
# optimizer step (gradient descent) 

# going from raw logits -> prediciton probabiities -> prediction labels
# we can conver these logits into predicition probabilities by passing them through a sigmoid activation function 
# sigmoid for binary cross estropy and softmax for multicalss classification
# the we can conver models predction probabilities into prediction labels by rounding them to 0 or 1 or taking the argmax 

# view the first 5 outputs of the foward pass on the test data
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]

#print(y_logits)
#print(y_test)

# use the sigmoid activation fucntion on our model logits to turn them into predition probabilities
y_pred_probs = torch.sigmoid(y_logits)
#print(y_pred_probs)
# for our prediction probability values, we need to perform a range-style rounding on them
# y_pred_probs >= 0.5, y = 1 (class 1)
# y_pred_probs < 0.5, y = 0 (class 0)
# find the predicted labels based on our prediciton probabilities
y_preds = torch.round(y_pred_probs).to(device)

# in full (logits -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# get rid of extra dimension
y_preds.squeeze()

# building a training and a testing loop
torch.manual_seed(42)
torch.mps.manual_seed(42)

# set epochs number
epochs = 1000

# put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# build training and evaluating loop
for epoch in range(epochs):
    # training
    model_0.train()

    # forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labels

    # calculate accuracy/loss #
    #loss = loss_fn(torch.sigmoid(y_logits), # nn.BCELoss expects prediction probabilities as input
    
    loss = loss_fn(y_logits, # nn.BCEWithLogitsLoss() expects logits as input
                   y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    # optimizer zero grad
    optimizer.zero_grad()

    # backpropagation
    loss.backward()

    # gradient descent
    optimizer.step()

    # testing
    model_0.eval()
    with torch.inference_mode():
        # forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # calculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
        # print out waths happening
        #if epoch % 10 == 0:
        #    print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Acc loss: {acc:.5f} | Test loss: {test_loss:.2f} | Test acc: {test_acc:.2f}%")

# make predictions and evaluate the model
"""import requests
from pathlib import Path

# download helper function from lear pytorch repo
if Path("helper_function.py").is_file():
    print("helper_function.py exists, skipping download")
else:
    print("Downloading helper_function.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/helpers/helper_functions.py")
    with open("./helper_function.py", "w") as f:
        f.write(request.text)"""

from helper_functions import plot_predictions, plot_decision_boundary
# plot decision boundary of the model
#plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
#plt.title("Train")
#plot_decision_boundary(model_0, X_train, y_train)       # pass in the training labels
#plt.subplot(1, 2, 2)
#plt.title("Test")
#plot_decision_boundary(model_0, X_test, y_test)   # pass in the model
#plt.show()

# improving a model (from a model perspective)
# add more layers
# add more hidden units - go from 5 to 10 hidden units
# fit for longer
# changing the activation functions
# change the learning rate
# change the loss function
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    
    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = CircleModelV1().to(device)
#print(model_1)

# create a loss function
loss_fn = nn.BCEWithLogitsLoss()

# create an optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                             lr=0.01)

# write a training and evaluation loop
torch.manual_seed(42)
torch.mps.manual_seed(42)

# train for longer
epochs = 0 #1000

# put data on the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# build training and evaluating loop
for epoch in range(epochs):
    # training
    model_1.train()

    # forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probabilities -> pred labels

    # calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # optimizer zero grad
    optimizer.zero_grad()

    # loss backwards (backpropagation)
    loss.backward()

    # optimizer stop (gradient descent)
    optimizer.step()

    # testing
    model_1.eval()

    with torch.inference_mode():
        # forward pass
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # calculate loss/accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        # print out what's happening
        #if epoch % 100 == 0:
        #    print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Train acc: {acc:.5f}% | Test loss: {test_loss:.2f} | Test acc: {test_acc:.2f}%")

# plot the decision boundary
# plot decision boundary of the model
#plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
#plt.title("Train")
#plot_decision_boundary(model_1, X_train, y_train)       # pass in the training labels
#plt.subplot(1, 2, 2)
#plt.title("Test")
#plot_decision_boundary(model_1, X_test, y_test)   # pass in the model
#plt.show()

# preparing data to see if our model can fit a straight line
# create some data
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

# create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias # linear regression formula without epsilon

# check the data
#print(len(X_regression))
#print(X_regression[:5], y_regression[:5])

# create train and test splits
train_split = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# check the lengths of each
#print(len(X_train_regression), len(y_train_regression), len(X_test_regression), len(y_test_regression))

#plot_predictions(X_train_regression, y_train_regression, X_test_regression, y_test_regression)


# adjusting model_1 tof fit a traight line
# same architecture as model_1 (but using nn.Sequential)
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

# loss and optimizer
loss_fn = nn.L1Loss() # MAE loss with regression data
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.01)

# train model
torch.manual_seed(42)
torch.mps.manual_seed(42)

# set the number of epochs
epochs = 0 #1000

# put the data on the target device
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

# training
for epoch in range(epochs):
    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred, y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # testing
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred, y_test_regression)

    # print out whats happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Test loss: {test_loss:.5f}")

# turn on evaluation mode
model_2.eval()

# make predictions (inference mode)
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

# plot data and predictions
#plot_predictions(train_data=X_train_regression.to("cpu"),
#                 train_labels=y_train_regression.to("cpu"),
#                 test_data=X_test_regression.to("cpu"), 
#                 test_labels=y_test_regression.to("cpu"),
#                 predictions=y_preds.to("cpu"))

# recreating non-linear data (red and blues circles)
# make and plot data
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
#plt.show()

# convert data to tensror and then to train and test splits
import torch
from sklearn.model_selection import train_test_split

# turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# building a model with non-linearity
from torch import nn
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=5) # fenomeno dectectado
        self.layer_3 = nn.Linear(in_features=5, out_features=1)  # 10 out_features na ultima camada nao performa tao bem quanto 5
        self.relu = nn.ReLU() # non-linear activation function

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    
model_3 = CircleModelV2().to(device)

# setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_3.parameters(), lr=0.1)

# training a model with non-linearity
torch.manual_seed(42)
torch.mps.manual_seed(42)

# put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# loop through data
# set epochs
#epochs = 1000

for epoch in range(epochs):
    # Training
    model_3.train()

    # Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilites -> prediction labels

    # calculate loss
    loss = loss_fn(y_logits, y_train) # BCEWIthLogitsLoss expects logits as input
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)
    
    # Optimizer zero grad
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # step the optimizer (gradient descent)
    optimizer.step()

    # testing
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # print out whats happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Train acc: {acc:.5f}% | Test loss: {test_loss:.5f} | Test acc: {test_acc:.5f}%")

# evaluating a model trained with non-linear activation functions
# make predictions
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

# plot decisions boundaries
#plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
#plt.title("Train")
#plot_decision_boundary(model_1, X_train, y_train)
#plt.subplot(1, 2, 2)
#plt.title("Test")
#plot_decision_boundary(model_3, X_test, y_test)
#plt.show()

# replicating non-linear activation functions
# create a tensor
A = torch.arange(-10, 10, 1, dtype=torch.float)

# visualize the tensor
#plt.plot(torch.relu(A))
#plt.show()

def relu(x: torch.tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0), x) # inputs must be tensors

# plot ReLU activation function
#plt.plot(relu(A))

# doing the same for sigmoid
def sigmoid(x: torch.tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

#plt.plot(torch.sigmoid(A))
#plt.show()