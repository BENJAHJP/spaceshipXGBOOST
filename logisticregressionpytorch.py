import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

data = pd.read_csv("/home/b3njah/Downloads/machine learning data sets/train.csv")
test1 = pd.read_csv("/home/b3njah/Downloads/machine learning data sets/test.csv")

test_before = pd.DataFrame(test1)
test = pd.DataFrame(test1)
df = pd.DataFrame(data)

df.isnull().sum()
df = df.dropna()

test['Age'] = test['Age'].fillna(test['Age'].mean())
test['HomePlanet'] = test['HomePlanet'].fillna(test['HomePlanet'].mode()[0])
test['CryoSleep'] = test['CryoSleep'].fillna(test['CryoSleep'].mode()[0])
test['Destination'] = test['Destination'].fillna(test['Destination'].mode()[0])
test['VIP'] = test['VIP'].fillna(test['VIP'].mode()[0])
test['RoomService'] = test['RoomService'].fillna(test['RoomService'].mean())
test['FoodCourt'] = test['FoodCourt'].fillna(test['FoodCourt'].mean())
test['ShoppingMall'] = test['ShoppingMall'].fillna(test['ShoppingMall'].mean())
test['VRDeck'] = test['VRDeck'].fillna(test['VRDeck'].mean())

test = test.drop(['Cabin', 'Spa', 'Name'], axis=1)
df = df.drop(['Cabin', 'Spa', 'Name'], axis=1)

home_planet_test_dummies = pd.get_dummies(test['HomePlanet'])
cryo_sleep_test_dummies = pd.get_dummies(test['CryoSleep'], drop_first=True)
test['CryoSleep'] = cryo_sleep_test_dummies
vip_test_dummies = pd.get_dummies(test['VIP'], drop_first=True)
test['VIP'] = vip_test_dummies
destination_planet_test_dummies = pd.get_dummies(test['Destination'])

home_planet_dummies = pd.get_dummies(df['HomePlanet'])
cryo_sleep_dummies = pd.get_dummies(df['CryoSleep'], drop_first=True)
df['CryoSleep'] = cryo_sleep_dummies
vip_dummies = pd.get_dummies(df['VIP'], drop_first=True)
df['VIP'] = vip_dummies
destination_planet_dummies = pd.get_dummies(df['Destination'])
transported_dummies = pd.get_dummies(df['Transported'], drop_first=True)
df['Transported'] = transported_dummies

test = test.drop(['HomePlanet', 'Destination', 'PassengerId'], axis=1)
df = df.drop(['HomePlanet', 'Destination', 'PassengerId'], axis=1)

test_data = pd.concat([home_planet_test_dummies, destination_planet_test_dummies, test], axis=1)
train_data = pd.concat([home_planet_dummies, destination_planet_dummies, df], axis=1)

Scaler = StandardScaler()
feature_scale = ['FoodCourt', 'Age', 'ShoppingMall', 'VRDeck', 'RoomService']
train_data[feature_scale] = Scaler.fit_transform(train_data[feature_scale])

test_data[feature_scale] = Scaler.fit_transform(test_data[feature_scale])

X = train_data.drop(['Transported'], axis=1)
y = train_data[['Transported']]

INPUT_DIMS = X.columns.shape[0]

X = X.to_numpy()
y = y.to_numpy()
test_data = test_data.to_numpy()

X = torch.from_numpy(X.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32))
test_data = torch.from_numpy(test_data.astype(np.float32))


class LogisticRegressor(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(LogisticRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dims, output_dims)
        self.fc2 = nn.Linear(output_dims, 1)

    def forward(self, data_input):
        x = self.fc1(data_input)
        output = torch.sigmoid(self.fc2(x))
        return output


model = LogisticRegressor(INPUT_DIMS, 6)

criterion = torch.nn.BCELoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 2000

for epochs in range(num_epochs):
    prediction = model(X)
    loss = criterion(prediction, y)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"epoch {epochs+1} loss {loss.item():.4f}")

    with torch.no_grad():
        predictions = model(test_data).detach().numpy()
        predictions = predictions.reshape(4277)


submission = pd.DataFrame({
    'PassengerId': test_before['PassengerId'],
    'Transported': predictions.astype(bool)
})

submission.to_csv("log.csv", index=False)