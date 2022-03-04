from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import numpy as np

data = load_digits()

X = data.images
y = data.target

print(y)
X = data.images / 255

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_test.shape)
X_train.reshape(1437, 8, 8, 1)
X_test.reshape(360, 8, 8, 1)

print(y_test.shape)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, activation='relu', kernel_size=(2, 2)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='Softmax'))

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

print(model.evaluate(X_test, y_test))

prediction = model.predict(X_test)

print(np.argmax(prediction))