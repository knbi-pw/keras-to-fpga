import tensorflow
from keras.layers import BatchNormalization, LeakyReLU
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.models import Sequential


def create_model():
    H = W = 32
    input_stream = Input(shape=(H, W, 3))
    x = input_stream

    x = Conv2D(8, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=True, name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    output_stream = x
    return x
