
from pyexpat.errors import XML_ERROR_INVALID_TOKEN
from MiniTownData import *
import torch
import math
import numpy as npq
import pickle

#import test set
x_test,y_test = minitownTrain()
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test).unsqueeze(1)

#import train set
a_file = open("MiniTownTrainData.pkl", "rb")
x_train,y_train = pickle.load(a_file)
a_file.close()
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train).unsqueeze(1)

# if want to see an example
print(x_train[0])
print(y_train[0])

# shuffle and split train vs valid
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

x_shuffled,y_shuffled = unison_shuffled_copies(x_train,y_train)

split = int(len(x_train) * 0.9)
X_TRAIN = x_shuffled[:split]
Y_TRAIN = y_shuffled[:split]

X_VAL =x_shuffled[split:]
Y_VAL =y_shuffled[split:]


# define model arch
model = torch.nn.Sequential(
    torch.nn.Linear(4, 4),   # Wx + b
    torch.nn.ReLU(),         # max(0,x)
    torch.nn.Linear(4,1),    # Wx + b
)

# define hyper-params
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# set model to training mode
model.train()

# iterate through epochs 
for epoch in range(400):
    ValAcc = 0
    model.train()
    for i in range(len(X_TRAIN)):

        y_pred = model(X_TRAIN[i].float())
        loss = loss_fn(y_pred, Y_TRAIN[i].float())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    model.eval()
    for i in range(len(X_VAL)):
        if abs(model(X_VAL[i].float()).detach().numpy()[0] - Y_VAL[i]) <= 4:
            ValAcc += 1

    if epoch % 20 == 0:
        print("----------EPOCH:",epoch)
        print("Train Loss:",loss.item())
        print("Val Accuracy:", 100*ValAcc/len(X_VAL),"%")

# switch to eval mode and get test preds
model.eval()
preds = [model(i.float()).detach().numpy()[0] for i in x_test]

# plot test set against ground truth
import matplotlib.pyplot as plt

plt.plot(np.arange(len(preds)),preds,label = 'preds')
plt.plot(np.arange(len(y_test)),y_test,label = 'ground truth')
plt.legend()
plt.show()

