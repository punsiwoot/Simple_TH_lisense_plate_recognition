import tensorflow as tf
import tensorflow.keras.layers as nn

model_lisense_plate_correct = tf.keras.Sequential([
    # nn.Rescaling(scale=1./255, offset=0.0),
    nn.Conv2D(filters=32, kernel_size = (10,10), activation='relu', input_shape=(100, 200, 1)),
    nn.Conv2D(filters=16, kernel_size = (5,5), activation='relu'),
    nn.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None),
    nn.Flatten(),
    nn.Dense(30,activation='relu'),
    nn.Dense(10,activation="relu"),
    nn.Dense(1)
])
