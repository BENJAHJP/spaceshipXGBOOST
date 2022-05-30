import keras
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.applications import efficientnet
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import cross_val_score

labels = pd.read_csv("/home/b3njah/Downloads/submission sample/efficientlabels.csv")
IMAGE_SIZE = 224

# prepare the labels dataset
labels = pd.DataFrame(labels)
print(labels.columns)


def prepare_labels(data):
    # species_dummies = pd.get_dummies(data['species'])
    data = pd.get_dummies(data['individual_id'])
    # data = pd.concat([species_dummies, individual_id_dummies], axis=1)
    return data


y = prepare_labels(labels)

print(f"Y HEAD {y.head()}")
print(f"Y COLUMNS {y.columns}")
final_y = np.array(y)
print(f"final y SHAPE {final_y.shape}")
print(f"Y SHAPE {y}")


def class_indices(data):
    data1 = data.columns
    one_hot_encoder = LabelEncoder()
    classes_indices = one_hot_encoder.fit_transform(data1)
    print(f"classes indices {classes_indices}")
    return classes_indices


y_class_indices = class_indices(y)
print(f" y class indices {y_class_indices}")


def prepare_images():
    image_data = []
    os.chdir("/home/b3njah/Downloads/train/images")
    for images in os.listdir():
        image = cv2.imread(images)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imshow("image", image)
        cv2.waitKey(1)
        # image = image.reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
        image = image / 255
        image_data.append(image)
    return image_data


X = prepare_images()
X = np.array(X)
print(f"X SHAPE {X.shape}")
print(f"X index 0 {X[0]}")

# train_generate = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)
# X = train_generate.fit(X)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
NUM_CLASSES = 13
# input_layer = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# model creation
model = Sequential()
base_model = efficientnet.EfficientNetB0(weights="imagenet",
                                         include_top=False,
                                         input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                         )

model.add(base_model)
model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
# model.add(Dropout(.3))
model.add(Dense(128, activation='relu'))
# model.add(Dropout(.2))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

print(f"y final shape{y.shape}")
model.fit(X, y_class_indices, epochs=50)

# predictions = model.predict(X)
#
# print(cross_val_score(model, X, y))
#
# print(model.score(X, y))
#
# matrix = ConfusionMatrixDisplay.from_predictions(y, predictions)
# plt.show()