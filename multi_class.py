import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# set the hyperparameters for data creation
NUM_SAMPLES = 1000
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# create multi-class data
X_blob, y_blob = make_blobs(n_samples=NUM_SAMPLES,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5, # give the clusters a bit of variance
                            random_state=RANDOM_SEED)

# turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor) # long == int64

# split into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, # features
                                                                        y_blob, # labels
                                                                        test_size=0.2, # 20% of data for testing
                                                                        random_state=RANDOM_SEED)

# plot data
#plt.figure(figsize=(10, 7))
#plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
#plt.show()

# building a muilti-class classification model


device = "mps" if True else "cuda" if torch.cuda.is_available() else "cpu"

# build a muilti-class classification model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """
        Initializer multi-class classification model

        Args:
            input_features (int): the number of input features
            output_features (int): the number of output classes
            hidden_units (int): the number of hidden units between layers, default 8
        
        Returns:
        
        Example:
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
            )

    def forward(self, x):
        return self.linear_layer_stack(x)
    
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_true)) * 100


# create an instance of BlobModel and sent to the target device
# optimizer updates the 
model_4 = BlobModel(input_features=NUM_FEATURES, output_features=NUM_CLASSES).to(device)
#print(model_4(X_blob_train.to(device))[0].shape, NUM_CLASSES)
# create a loss function ofr muti-class classification
loss_fn = nn.CrossEntropyLoss()

# create an optimizer for multi-class classification
optimizer = torch.optim.SGD(params=model_4.parameters(), lr=0.1)

"""
    In order to evaluate and train and test our model, we need to convert our model's outputs (logits)
    to prediciton probabilites and the to prediction labels
    Logits (raw output of the model)-> Pred probs (use 'torch.softmax') -> Pred labels (take the argmax of the predicition probabilities)
"""
# Make prediction logits with model
y_logits = model_4(X_blob_test.to(device))
# Perform softmax calculation on logits across dimension 1 to get prediction probabilities

# getting predicitons probabilities for a multi-class model
# get some raw outputs of our model (logits)
#model_4.eval()
#with torch.no_grad():
#    y_logits = model_4(X_blob_test.to(device))

# convert our models logit outputs to prediciton probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)
#print(torch.sum(y_logits[0]))
#print(torch.argmax(y_pred_probs[0]))
#print((y_pred_probs[:5]))

# convert models prediction probabilities to prediction labels
y_preds = torch.argmax(y_pred_probs, dim=1)

# creating a traning and a testing loop for the multi-class model
# fit the multi-class model to the data
torch.manual_seed(42)

# set number of epocs
epochs = 100

# put data to the target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

# loop through data
for epoch in range(epochs):

    model_4.train()
    y_logits = model_4(X_blob_train)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_preds)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # testing
    model_4.eval()
    with torch.inference_mode():
        #forward pass
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        # calculate loss and accuracy
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_preds)

    # print results
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.2f} |  Acc: {acc:.2f}% | Test loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")

# making and evaluanting prediciton 
# make predictions with the multi-class model
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

# go from logits -> predictions probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)
#print(y_pred_probs[:5])
# go from pred probs to pred labels
y_pred_probs = torch.argmax(y_pred_probs, dim=1)
#print(y_preds[:10])
from helper_function import plot_decision_boundary


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
plt.show()

"""
A few more classification metrics:

accuracy - out of 100 samples, how many does our model get right?
Default metric for classification problems. Not the best for imbalanced classes

precision - out of all the positive classes our model predicted, how many were actually positive?\
Higher precision leads to less false positives.

recall - out of all the positive classes, how many did our model predict?
Higher recall leads to less false negatives.

F1-score - a combination of precision and recall, closer to 1.0 is better
Combination of precision and recall, usually overall metric for a classification model

confusion matrix - a table comparing the true labels to the predicted labels
When comparing predictions to truth labels to see where models get confused.
Can be hard to use with large number of classes

classification report - a collection of some of the main classification metrics such as precision, recall and F1-score
"""