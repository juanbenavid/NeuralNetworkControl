
from MiniTownData import *
import torch
import math
import numpy as np

x_train,y_train = minitownTrain()

x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)

print(x_train[0])
print(y_train[0])

model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8,4),
    torch.nn.ReLU(),
    torch.nn.Linear(4,1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()

for epoch in range(200):
    for i in range(len(x_train)):

        y_pred = model(x_train[i].float())
        loss = loss_fn(y_pred, y_train[i].float())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    if epoch % 10 == 0:
        print(epoch, loss.item())

model.eval()
preds = [model(i.float()).detach().numpy()[0] for i in x_train]

import matplotlib.pyplot as plt

plt.plot(np.arange(len(preds)),preds,label = 'preds')
plt.plot(np.arange(len(y_train)),y_train,label = 'ground truth')
plt.legend()
plt.show()

