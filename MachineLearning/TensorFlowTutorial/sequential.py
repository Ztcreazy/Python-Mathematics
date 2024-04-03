import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.datasets import mnist
from keras import layers

from tensorflow import keras

import tensorflow as tf
# import keras

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x train shape: ", x_train.shape)
print("y train shape: ", y_train.shape)

x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
print("x train shape: ", x_train.shape)

x_test  = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# Sequential
model = keras.Sequential(
    [
        keras.Input(shape=(28*28, )),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)

print("model 1 summary: ", model.summary())

model2 = keras.Sequential()
model2.add(keras.Input(shape=(784, ), name="input_layer_1"))
model2.add(layers.Dense(512, activation="relu", name="hidden_layer_1"))
model2.add(layers.Dense(256, activation="relu", name="hidden_layer_2")) # 
model2.add(layers.Dense(10, name="output_layer_1"))

print("model 2 summary: ", model2.summary())

# model = keras.Model(inputs = model.inputs, # inputs !!!!
                    # outputs = [model.layers[-2].output]) # layers !!!!
                    # outputs = [model.get_layer('my_layer').output]
                    # outputs = [layer.output for layer in model.layers]
# feature = model.predict(x_train)
# for feature in features:
# print(feature.shape) 
# print("feature shape: ", feature.shape)

# import sys
# sys.exit()

from keras.optimizers import SGD
from keras.optimizers import Adagrad

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # optimizer = keras.optimizers.Adam(learning_rate=0.001),
    # optimizer = SGD(learning_rate=0.01, momentum=0.9),
    optimizer = Adagrad(learning_rate=0.01, initial_accumulator_value=0.1, epsilon=1e-7),
    metrics = ["accuracy"],
)

model.fit(x_train, y_train, batch_size = 32, epochs = 5, verbose = 2)
model.evaluate(x_test, y_test, batch_size = 32, verbose = 2)

