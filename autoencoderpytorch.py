import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dataset = load_digits()
X = dataset.images
X = X / 255

print("ARGMAX", np.argmax(X))
print(X)
X = torch.from_numpy(X)
X = X.to(torch.float32)
# mean = X[0].mean()
# std = X[0].std()
#
# transform = transforms.Normalize(mean=mean, std=std)
# X = transform(X)

X = X.view(-1, 8*8)

print(X.shape)
print("TORCH MAX", torch.max(X))


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8*8),
            nn.Sigmoid()
        )

    def forward(self, data):
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencoder()
# for name, param in model.named_parameters():
#     print(name, param.shape)
# hyper parameters
learning_rate = 0.01
weight_decay = 1e-5
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train out model
num_epochs = 100

for epoch in range(num_epochs):
    # get loss
    predicted = model(X)
    loss = criterion(predicted, X)

    # backward propagate
    loss.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f"epochs {epoch+1} loss {loss.item():.4f}")

prediction = predicted.detach().numpy()
X = X.detach().numpy()
for img in prediction:
    img = img.reshape(-1, 8, 8)
    X = X.reshape(-1, 8, 8)
    plt.figure(figsize=(8, 2))
    plt.gray()
    plt.imshow(img[0])

    plt.show()
