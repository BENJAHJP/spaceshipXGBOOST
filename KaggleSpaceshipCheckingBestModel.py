import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("/home/b3njah/Downloads/machine learning data sets/train.csv")
test = pd.read_csv("/home/b3njah/Downloads/machine learning data sets/test.csv")

# data = pd.read_csv("/content/train.csv")
# test1 = pd.read_csv("/content/test.csv")

# convert data into dataframe

test = pd.DataFrame(test)
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

# drop Home-planet, destination, passengerId

test = test.drop(['HomePlanet', 'Destination', 'PassengerId'], axis=1)
df = df.drop(['HomePlanet', 'Destination', 'PassengerId'], axis=1)

# join data

test_data = pd.concat([home_planet_test_dummies, destination_planet_test_dummies, test], axis=1)
train_data = pd.concat([home_planet_dummies, destination_planet_dummies, df], axis=1)
# print(train_data.columns)
# print(test_data.columns)
# print(test_data['CryoSleep'])
print(test_data.head())
# scale FoodCourt, PassengerId, Age, ShoppingMall, VRDeck, RoomService

Scaler = StandardScaler()
feature_scale = ['FoodCourt', 'Age', 'ShoppingMall', 'VRDeck', 'RoomService']
train_data[feature_scale] = Scaler.fit_transform(train_data[feature_scale])

test_data[feature_scale] = Scaler.fit_transform(test_data[feature_scale])

# X data and y data
# print(train_data)
X = train_data.drop(['Transported'], axis=1)
y = train_data[['Transported']]

# convert to numpy and split dataset

X = X.to_numpy()
y = y.to_numpy()

# n_samples = len(train_data['Transported'])
# y = y.reshape(n_samples, -1)

y = y.ravel()
print("y shape is :", y.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# reshape dataset for feeding our model

# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
# y_train = y_train.reshape(y_train.shape[0], 1)
# y_test = y_test.reshape(y_test.shape[0], 1)


# create param
model_param = {
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(),
        'param': {
            'criterion': ['gini', 'entropy']
        }
    },
    'KNeighboursClassifier': {
        'model': KNeighborsClassifier(),
        'param': {
            'n_neighbors': [5, 10, 15, 20, 25, 30]
        }
    },
    'SVC': {
            'model': SVC(),
            'param': {
                'kernel': ['rbf', 'linear', 'sigmoid'],
                'C': [0.001, 0.1, 1, 10, 100, 1000]
            }
    }
}

# implement model selection
scores = []
for model_name, mp in model_param.items():
    model_selection = GridSearchCV(estimator=mp['model'], param_grid=mp['param'], cv=5, return_train_score=False)
    model_selection.fit(X, y)
    scores.append({
        'model': model_name,
        'best_score': model_selection.best_score_,
        'best_params': model_selection.best_params_
    })

    model_score = pd.DataFrame(scores)
    print("model score is : ", model_score)
