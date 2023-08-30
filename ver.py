# This model:
# Start with random values (weitghts and bias)
# Look at training data and adjust the random values to better represent the ideal values (weights and bias)
# Does ot through two main algorithms:
# 1 - Gradient descent
# 2 - Backpropagation

import torch
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


    pass