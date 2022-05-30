from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X = digits.images
y = digits.target

n_samples = len(X)
X = X.reshape(n_samples, -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

model = svm.SVC()
model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.score(X_test, y_test))


