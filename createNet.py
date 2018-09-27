#!/usr/bin/python3

import numpy as np
from tensorflow import keras

def main():
    prepareModel("valid") # no zero-padding
    prepareModel("same") # zero-padding


def prepareModel(paddingMode):
    model = createModel(paddingMode)
    printModelOutput(model, paddingMode)
    model.save(paddingMode + ".h5")


def createModel(paddingMode):
    # Create layers
    bn = keras.layers.BatchNormalization(beta_initializer="ones") # Initialized to bn(x) == x + 1
    conv = keras.layers.Convolution2D(1, (3, 3), padding=paddingMode,
                                      kernel_initializer="ones", bias_initializer="zeros")

    # Define net (that initializes weights)
    x = inp = keras.layers.Input(shape=(5, 5, 1))
    x = bn(x)
    x = conv(x)

    # Manipulate weights
    model = keras.models.Model(inputs=inp, outputs=x)
    return model


def printModelOutput(model, paddingMode):
    inp = np.zeros((1, 5, 5, 1), dtype=np.float32)
    out = model.predict(inp)
    print()
    print(paddingMode + ":")
    print("input:")
    print(inp)
    print()
    print("output:")
    print(out)
    print()


if __name__ == "__main__":
    main()
