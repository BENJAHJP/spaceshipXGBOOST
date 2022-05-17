import torch
import torch.nn as nn
from sklearn.datasets import load_iris

dataset = load_iris()
X = dataset.data
y = dataset.target

X = torch.from_numpy(X)
X = X.to(torch.float32)
y = torch.from_numpy(y)
y = y.to(torch.float32)
y = y.view(y.shape[0], -1)


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.log = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x = self.log(data)
        return x


model = LogisticRegression()
print(model)
# hyper parameters
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training
num_epochs = 100

for epoch in range(num_epochs):
    # loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # loss backward prop
    loss.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f" epoch {epoch+1} loss {loss.item():.4f}")
