# Conversion script for tiny-yolo-voc to Metal.
# Needs Python 3 and Keras 1.2.2

import os
import numpy as np
import keras
from keras.models import Sequential, load_model

model_path = "yad2k/model_data/tiny-yolo-voc.h5"
dest_path = "../TinyYOLO-NNGraph/Parameters"

# Load the model that was exported by YAD2K.
model = load_model(model_path)
model.summary()

print("\nConverting parameters...")

def export_conv_and_batch_norm(conv_layer, bn_layer, name):
    print(name)

    # Keras stores the learnable weights for a BatchNormalization layer
    # as four separate arrays:
    #   0 = gamma (if scale == True)
    #   1 = beta (if center == True)
    #   2 = moving mean
    #   3 = moving variance
    bn_weights = bn_layer.get_weights()
    gamma = bn_weights[0]
    beta = bn_weights[1]
    mean = bn_weights[2]
    variance = bn_weights[3]

    # Get the weights for the convolution layer and transpose from
    # Keras order to Metal order.
    conv_weights = conv_layer.get_weights()[0]
    conv_weights = conv_weights.transpose(3, 0, 1, 2).flatten()

    # We're going to save the conv_weights and the BN parameters
    # as a single binary file.
    combined = np.concatenate([conv_weights, mean, variance, gamma, beta])
    combined.tofile(os.path.join(dest_path, name + ".bin"))

def export_conv(conv_layer, name):
    print(name)
    conv_weights = conv_layer.get_weights()[0]
    conv_weights = conv_weights.transpose(3, 0, 1, 2)
    conv_weights.tofile(os.path.join(dest_path, name + ".bin"))

export_conv_and_batch_norm(model.layers[1], model.layers[2], "conv1")
export_conv_and_batch_norm(model.layers[5], model.layers[6], "conv2")
export_conv_and_batch_norm(model.layers[9], model.layers[10], "conv3")
export_conv_and_batch_norm(model.layers[13], model.layers[14], "conv4")
export_conv_and_batch_norm(model.layers[17], model.layers[18], "conv5")
export_conv_and_batch_norm(model.layers[21], model.layers[22], "conv6")
export_conv_and_batch_norm(model.layers[25], model.layers[26], "conv7")
export_conv_and_batch_norm(model.layers[28], model.layers[29], "conv8")
export_conv(model.layers[31], "conv9")

print("Done!")
