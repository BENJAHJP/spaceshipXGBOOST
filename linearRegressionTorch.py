import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt

# dataset
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# convert numpy into torch tensor


def numpy_to_torch(data):
    data = torch.from_numpy(data.astype(np.float32))
    return data


X = numpy_to_torch(X_numpy)
y = numpy_to_torch(y_numpy)
# reshape y
y = y.view(y.shape[0], 1)

# model
n_samples, n_features = X.shape
print(X.shape)
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
num_epochs = 300

for epochs in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    # backward pass
    loss.backward()
    # weight update
    optimizer.step()
    optimizer.zero_grad()

    if (epochs+1) % 10 == 0:
        print(f"---------- epochs {epochs+1} loss {loss.item():.4f} ----------")

# plot data
predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()





