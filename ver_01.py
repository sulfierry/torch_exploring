import torch
from torch import nn
import matplotlib.pyplot as plt


device = "mps" if True else "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using device: {device}")

# create some data using the linear regression formula of y = weight * X + bias
weight = 0.7
bias = 0.3

# create range values
start = 0
end = 1
step = 0.02

# create x features and y labels
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
#print(len(X_train), len(X_test), len(y_train), len(y_test))

#plot_predictions(X_train, y_train, X_test, y_test)

# create a linear model by subclassing nn.Module
class LinearRegrassionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    # define the forward method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
# set the manual seed
torch.manual_seed(42)
model_1 = LinearRegrassionModel()
#print(model_1, model_1.state_dict())

# check the model current device
#next(model_1.parameters()).device

# set the model to use the target device (cuda)
#model_1.to(device)
#next(model_1.parameters()).device

# Training, we need:
# Loss function
# Optimizer
# Training loop
# Testing loop

# set up loss function
loss_fn = nn.L1Loss()

# setup optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

# write a training loop
torch.manual_seed(42)

# put data on the target device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
  
epochs = 200

for epoch in range(epochs):
    # set the model to training mode
    model_1.train()
    

    # compute the predictions
    y_pred = model_1(X_train)
    
    # compute the loss
    loss = loss_fn(y_pred, y_train)
    
    # zero the gradients
    optimizer.zero_grad()
    
    # backpropagate the loss
    loss.backward()
    
    # update the parameters
    optimizer.step()
    
    # Testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)
        

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss}, | Test loss: {test_loss}") 


# create a model
def plot_predictions(train_data=X_train.cpu(),
                     train_labels=y_train.cpu(),
                     test_data=X_test.cpu(),
                     test_labels=y_test.cpu(),
                     predicitions=None):
    
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing data')
    if predicitions is not None:
        plt.scatter(test_data, predicitions, c='r', s=4, label='Predictions')
    plt.legend(prop={'size': 14})
    plt.show()

# Turn model into evaluation mode
model_1.eval()

# make predictions on the test data
with torch.inference_mode():
    y_preds = model_1(X_test)

print(y_preds)

#y_preds = y_preds.cpu()
# plot the predictions
plot_predictions(predicitions=y_preds.cpu())

from pathlib import Path
# create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2 create model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(model_1.state_dict(), MODEL_SAVE_PATH)

# Load a PyTorch
# create new insatnce of the model
loaded_model_1 = LinearRegrassionModel()

# load the saved model_1 state dict
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# put the loaded model to device
print(loaded_model_1.to(device))
print(next(loaded_model_1.parameters()).device)
print(loaded_model_1.state_dict())


# evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)

print(y_preds == loaded_model_1_preds)