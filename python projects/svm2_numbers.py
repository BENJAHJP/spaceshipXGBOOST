import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

X = digits.images
y = digits.target

n_samples = len(X)
X = X.reshape(n_samples, -1)
print(f"X shape : {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model = svm.SVC()

model.fit(X_train, y_train)

print(f"Model prediction : {model.predict(X_test)}")

print(f"Model score : {model.score(X_test, y_test)}")