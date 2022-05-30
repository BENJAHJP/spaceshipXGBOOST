import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import keras
from keras.layers import Dense
from keras.models import Sequential

# data = pd.read_csv("/home/b3njah/Downloads/machine learning data sets/train.csv")
# test = pd.read_csv("/home/b3njah/Downloads/machine learning data sets/test.csv")

data = pd.read_csv("/content/train.csv")
test1 = pd.read_csv("/content/test.csv")

# convert data into dataframe

test = pd.DataFrame(test1)
df = pd.DataFrame(data)

# check data features

print(df.columns)

# check for correlation

print(spearmanr(df["FoodCourt"], df["Transported"]))

# check for null values and drop rows with at least one null values

df.isnull().sum()
df = df.dropna()

# print("test.isnull().sum()", test.isnull().sum())
# check for null and engineer the features

test['Age'] = test['Age'].fillna(test['Age'].mean())
test['HomePlanet'] = test['HomePlanet'].fillna(test['HomePlanet'].mode()[0])
test['CryoSleep'] = test['CryoSleep'].fillna(test['CryoSleep'].mode()[0])
test['Destination'] = test['Destination'].fillna(test['Destination'].mode()[0])
test['VIP'] = test['VIP'].fillna(test['VIP'].mode()[0])
test['RoomService'] = test['RoomService'].fillna(test['RoomService'].mean())
test['FoodCourt'] = test['FoodCourt'].fillna(test['FoodCourt'].mean())
test['ShoppingMall'] = test['ShoppingMall'].fillna(test['ShoppingMall'].mean())
test['VRDeck'] = test['VRDeck'].fillna(test['VRDeck'].mean())

# drop cabin and spa

test = test.drop(['Cabin', 'Spa', 'Name'], axis=1)
df = df.drop(['Cabin', 'Spa', 'Name'], axis=1)

# one hot encode home-planet, cryo-sleep, VIP, Transported

home_planet_test_dummies = pd.get_dummies(df['HomePlanet'])
test['CryoSleep'] = test['CryoSleep'].apply(lambda x: 1 if x == 'True' else 0)
test['VIP'] = test['VIP'].apply(lambda x: 1 if x == 'True' else 0)
destination_planet_test_dummies = pd.get_dummies(test['Destination'])

home_planet_dummies = pd.get_dummies(df['HomePlanet'])
df['CryoSleep'] = df['CryoSleep'].apply(lambda x: 1 if x == 'True' else 0)
df['VIP'] = df['VIP'].apply(lambda x: 1 if x == 'True' else 0)
destination_planet_dummies = pd.get_dummies(df['Destination'])
df['Transported'] = df['Transported'].apply(lambda x: 1 if x == 'True' else 0)

# drop Home-planet, destination, passengerId

test = test.drop(['HomePlanet', 'Destination', 'PassengerId'], axis=1)
df = df.drop(['HomePlanet', 'Destination', 'PassengerId'], axis=1)

# join data

test_data = pd.concat([home_planet_test_dummies, destination_planet_test_dummies, test], axis=1)
train_data = pd.concat([home_planet_dummies, destination_planet_dummies, df], axis=1)
print(train_data.columns)
print(test_data.columns)


# scale FoodCourt, PassengerId, Age, ShoppingMall, VRDeck, RoomService

Scaler = StandardScaler()
feature_scale = ['FoodCourt', 'Age', 'ShoppingMall', 'VRDeck', 'RoomService']
train_data[feature_scale] = Scaler.fit_transform(train_data[feature_scale])

test_data[feature_scale] = Scaler.fit_transform(test_data[feature_scale])

# X data and y data
X = train_data.drop(['Transported'], axis=1)
y = train_data['Transported']

# convert to numpy and split dataset

X = X.to_numpy()
y = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# reshape dataset for feeding our model

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# create model

model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile

model.compile(loss=keras.losses.BinaryCrossentropy(),
              optimizer='sgd',
              metrics=['Accuracy'])

# fit

model.fit(X_train, y_train, epochs=30, verbose=True)

# evaluate model

model.evaluate(X_test, y_test)

# model predictions

predictions = model.predict(test_data)

# reshape

predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])

# convert to dataframe

predictions = pd.DataFrame(predictions)

predictions.apply(lambda x: 1 if x >= 0.0005 else 0)