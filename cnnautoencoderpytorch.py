import torch
import torch.nn as nn
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

dataset = load_digits()
X = dataset.images
X = X / 255

X = torch.from_numpy(X)
X = X.to(torch.float32)
X = X.view(-1, 1, 8, 8)

print(X.shape)


class CnnAutoEncoder(nn.Module):
    def __init__(self):
        super(CnnAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=2, output_padding=1),
            #nn.Sigmoid()
        )

    def forward(self, data_input):
        encoded = self.encoder(data_input)
        decoded = self.decoder(encoded)
        return decoded


model = CnnAutoEncoder()

print(model)

# hyper parameters
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# training
num_epochs = 100
outputs = []
for epoch in range(num_epochs):
    # loss
    reconstructed = model(X)
    loss = criterion(reconstructed, X)

    # backward propagation
    loss.backward()

    # optimize weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f"epoch {epoch+1} loss {loss.item():.4f}")
        outputs.append((epoch, X, reconstructed))

# for k in range(0, num_epochs, 4):
#     plt.figure(figsize=(8, 2))
#     plt.gray()
#     img = outputs[k][1].detach().numpy()
#     reconstructed = outputs[k][2].detach().numpy()
#     for i, item in enumerate(img):
#         if i >= 9:
#             break
#         plt.subplot(2, 9, i+1)
#         plt.imshow(item[0])
#
#     for i, item in enumerate(reconstructed):
#         if i >= 9:
#             break
#         plt.subplot(2, 9, 9+i+1)
#         plt.imshow(item[0])

prediction = reconstructed.detach().numpy()
X = X.detach().numpy()

for img in prediction:
    plt.gray()
    plt.imshow(img[0])
    plt.show()
