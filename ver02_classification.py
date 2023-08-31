# Mkae classification data and get it ready
from sklearn.datasets import make_circles

# make samples
n_samples = 1000
# create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

#print(len(X), len(y))
#print(f"First 5 samples of X:\n {X[:5]}")
#print(f"First 5 samples of y:\n {X[:5]}")

# make dataframe of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0], 
                        "X2": X[:, 1], 
                        "label": y})


# visualize
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], 
            X[:, 1], 
            c=y, 
            cmap=plt.cm.RdYlBu)
#plt.show()

# view first example of features and labels
X_sample = X[0]
y_sample = y[0]

print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Values for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

# Turn data into tensors
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

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
# visualize the results

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
print(model_0)

# replicate the model above using nn.Sequential()
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

print(model_0)