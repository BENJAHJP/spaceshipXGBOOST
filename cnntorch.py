import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

data = load_digits()
X = data.images
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


def convert_to_torch(data_input):
    data_input = torch.from_numpy(data_input.astype(np.float32))
    return data_input


X_train = convert_to_torch(X_train)
X_test = convert_to_torch(X_test)
y_train = convert_to_torch(y_train)
y_test = convert_to_torch(y_test)


def reshape_y(y_input):
    y_input = y_input.view(y_input.shape[0], 1)
    return y_input


y_train = reshape_y(y_train)
y_test = reshape_y(y_test)

X_train = X_train.reshape(1437, 1, 8, 8)

# one hot encode y_train


def onehotencode(y_data):
    y_data = F.one_hot(y_data.to(torch.int64))
    y_data = y_data.to(torch.float32)
    y_data = y_data.reshape(y_data.shape[0], 10)
    return y_data


y_train = onehotencode(y_train)
y_test = onehotencode(y_test)

print(y_train)
# dataloader
# create model


class ConvolutionalNeuralNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvolutionalNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.fc1 = nn.Linear(64*2*2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, data_input):
        x = F.relu(self.conv_layer1(data_input))
        x = self.max_pool1(x)
        x = F.relu(self.conv_layer2(x))
        x = self.max_pool1(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


model = ConvolutionalNeuralNet()
print(model)

# Hyper parameters
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train model
num_epochs = 300

for epoch in range(num_epochs):
    # get loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    # backpropagation
    loss.backward()
    # optimize weights
    optimizer.step()
    optimizer.zero_grad()

    if(epoch+1) % 5 == 0:
        print(f"------ epoch {epoch+1} loss {loss.item():.4f} accuracy --------")

