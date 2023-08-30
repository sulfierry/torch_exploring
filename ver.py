# This model:
# Start with random values (weitghts and bias)
# Look at training data and adjust the random values to better represent the ideal values (weights and bias)
# Does ot through two main algorithms:
# 1 - Gradient descent
# 2 - Backpropagation

import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


# known paremeters
weight = 0.7
bias = 0.3

# create
start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# creaste a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# create a model
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predicitions=None):
    
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing data')
    if predicitions is not None:
        plt.scatter(test_data, predicitions, c='r', s=4, label='Predictions')
    plt.legend(prop={'size': 14})
    plt.show()

#plot_predictions()
# linear regression model class
class LinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1,requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float32))
        
    # foward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is a input
        return self.weights * x + self.bias # linear regression formula

# torch.nn - contains all of the buildings for computational graphs
# torch.nn.Parameter - what parameters should be adjusted during training
# torch.nn.Module - The base class for all neural network modules, if you subclass it, you should overwrite forward()
# torch.optim - this is a package implementing various optimization algorithms (help with gradient descent)
# def foward() - All nn.Module subclasses override the forward() method, which defines the computation performed at every call
# torch.utils.data.Dataset - Represents a map between key (label) and sample (features) pairs of your data. Such as images and their associated labels
# torch.utils.data.DataLoader - Allows you to iterate over your data

# create a random seed
torch.manual_seed(42)

# create a instance of the model (this is a subclass of nn.Module)
model_0 = LinearRegressionModel()

# a parameter is a value that the model sets itself

#print(list(model_0.parameters()))
#print(model_0.state_dict())

# making prediction using 'torch.inference_mode()'
# To check the model predictive power, we need see how well it predicts 'y_test' based on 'X_test'
# When we pass data through the model, its going to run it through the forward() method

# Make predictions with model
with torch.inference_mode():
    preds = model_0(X_test)
    y_preds = model_0(X_test)


#plot_predictions(predicitions=y_preds)

# Train model
# The whole idea of training is for a model to move from some unknown parameters to some known parameters
# one way to mesure how poor or good a model is, is by using a loss function
# Loss function - is a way to measure how wrong the models predictions are. To ideal outputs, lower is better
# Optimizer - Takes into account the loss of a model and adjust the models parameters (weight & bias) to improve the loss function
# And scpecifically for PyTorch we need a training and a testing loop
print(list(model_0.parameters()))


# setup a loss function
loss_fn = nn.L1Loss()

# setup an optmizer
optimizer = torch.optim.SGD(
    params=model_0.parameters(), lr=0.01) # lr - learning rate (most important hyperparameter)

# build a training  and testing loop
# 0 Loop through the data
# 1 Forward pass (this involver data moving throug the model) to make predictions - also called forward propagation
# 2 Calculate the loss (compare foward pass predictions to ground truth labels)
# 3 Optomezer zero grad
# 4 Loss backward (backpropagation) - move backwards through the network to calculate the gradients of each parameters of the model with respect to the loss
# 5 Optimzer setup (gradient descent) - use the optimizer to adjust our model parameters to try and improve the loss


## Training
# An epoch is one loop through the entire dataset
model_0.parameters()
epochs = 200

# to track these values
epoch_count = []
loss_values = []
test_loss_value = []

# 0 Loop through the data
for epoch in range(epochs):
    # set the model to training mode
    model_0.train() # train mode in PyTorch set all parameters that requires gradients 

    # 1 Foward pass
    y_pred = model_0(X_train)

    # 2 Caulculate the loss
    loss = loss_fn(y_pred, y_train)
    #print(f"Loss: {loss}")

    # 3 Optimizer zero grad
    optimizer.zero_grad()

    # 4 Loss backward
    loss.backward()

    # 5 Optimizer step
    # By default how the optimizer changes will accumulate through the loop
    # so we have to zero them above in step 3 for the next iteration of the loop
    optimizer.step()

    # TESTING ##############################################################################################
    # turn off different settings in the model not needed for evaluation/testing (dropout/batch norm layers)
    model_0.eval() # turn off gradient tracking

    with torch.inference_mode():
        # 1 - Do the foward model pass
        test_pred = model_0(X_test)

        # 2 - Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    # print out whats happening
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_value.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")


# plot the loss curves
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label='Train loss')
plt.plot(epoch_count, test_loss_value, label='Test loss')
plt.title("Training and test loss curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

with torch.inference_mode():
    y_preds_new = model_0(X_test)
    plot_predictions(predicitions=y_preds_new)