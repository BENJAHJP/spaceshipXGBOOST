import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as alb
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df1 = pd.read_csv("ABCD.csv")

# prepare labels
def prepare_labels(dataframe):
    # drop null values
    dataframe = dataframe.dropna()
    # get dummies for cell type
    cell_type_dummies = pd.get_dummies(dataframe['cell_type'], drop_first=True)
    dataframe = dataframe.drop(['localization', 'cell_type'], axis=1)
    dataframe = pd.concat([dataframe, cell_type_dummies], axis=1)
    return dataframe

df1 = prepare_labels(df1)

# create a class to handle our dataset
class BenignDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[index, 0])
        image = io.imread(img_path)
        y = torch.tensor(int(self.dataframe.iloc[index, 1:]))

        if self.transform:
            image = self.transform(image)
        return image, y

# data augmentation to reduce overfitting
dataset_transforms = alb.Compose([
        alb.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        alb.RandomCrop(height=256, width=256),
        alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        transforms.ToTensor()
    ]
)


batch_size=32
num_epochs = 100
learning_rate = 0.001

dataset = BenignDataset(df1, "benign_data", transform=dataset_transforms)

train_set, test_set, val_set = torch.utils.data.random_split(dataset=dataset, [4000, 687, 686])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

class ConvolutionalNeuralNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvolutionalNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        self.fc1 = nn.Linear(64*2*2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, data_input):
        x = F.relu(self.conv_layer1(data_input))
        x = self.max_pool1(x)
        x = F.relu(self.conv_layer2(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv_layer3(x))
        x = self.max_pool1(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))


model = ConvolutionalNeuralNet()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    losses = []
    for i, (data, target) in enumerate(train_loader):
        # run on gpu if available
        data = data.to(device)
        target = target.to(device)

        # train model forward pass
        scores = model(data)
        loss = criterion(scores, target)

        # backward propagation
        losses.append(loss.item())
        loss.backward()

        # update the weights
        optimizer.step()
        optimizer.zero_grad()

        print(f"epoch{epoch+1} loss {sum(losses)/len(losses)}")

def accuracy(data_loader, model):
    num_samples = 0
    num_correct = 0
    model.eval()

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            score = model(X)
            _, predictions = score.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f"correct {num_correct} / {num_samples} accuracy {float(num_correct)/float(num_samples)*100}")
    model.train()

print("accuracy on training set")
accuracy(train_loader, model)

print("accuracy on testing set")
accuracy(test_loader, model)
