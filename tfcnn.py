from keras.layers import Layer
import keras
from keras import layers, Model


class CnnBlock(Layer):
    def __init__(self):
        super(CnnBlock, self).__init__()
        self.cnn1 = layers.Convolution2D(16, 3, padding='same')
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2))
        self.cnn2 = layers.Convolution2D(32, 3, padding='same')
        self.cnn3 = layers.Convolution2D(64, 3, padding='same')

    def call(self, inputs, training=None, mask=None):
        x = self.cnn1(inputs)
        x = self.cnn2(x)
        x = self.cnn3(x)
        return self.pool1(x)


class Cnn(Model):
    def __init__(self, out_channels, kernel_size):
        super(Cnn, self).__init__()
        self.cov1 = layers.Convolution2D(out_channels, kernel_size, padding='same')
        self.block1 = CnnBlock()
        self.block2 = CnnBlock()

    def call(self, inputs, training=None, mask=None):
        x = self.cov1(inputs)
        x = self.block1(x)
        return self.block2(x)

    def get_config(self):
        pass

    def model(self):
        x = keras.Input(shape=(8, 8, 1))
        return keras.Model(inputs=x, outputs=self.call(x))


model = Cnn(8, 3)
model.build(input_shape=(1757, 8, 8, 1))
print(model.model().summary())