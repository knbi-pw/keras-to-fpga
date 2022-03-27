import numpy as np
from tensorflow import keras


def keras_weights_to_caffe_model(keras_model, caffe_model):
    conv_idx=0
    for layer in keras_model.layers:

        # skip if there is no caffe layer named accordingly
        if not np.any([x == layer.name for x in caffe_model._layer_names]):
            continue

        if type(layer) == keras.layers.Convolution2D:
            data = layer.get_weights()
            w, b = data if np.shape(data)[0] > 1 else [data[0], np.zeros(
                (1, np.shape(data)[-1]))]  # the bias term might not be present
            layer_name = layer.name
            caffe_model.params[layer_name][0].data[...] = np.transpose(w,(3, 2, 0, 1))  # Caffe wants (c_out, c_in, h, w)
            caffe_model.params[layer_name][1].data[...] = b
            conv_idx+=1

        if type(layer) == keras.layers.BatchNormalization:
            gamma, beta, mean, variance = layer.get_weights()
            caffe_model.params[layer.name][0].data[...] = mean
            caffe_model.params[layer.name][1].data[...] = variance + 1e-3
            caffe_model.params[layer.name][2].data[...] = 1  # always set scale factor to 1
            caffe_model.params['{}_sc'.format(layer.name)][0].data[...] = gamma  # scale
            caffe_model.params['{}_sc'.format(layer.name)][1].data[...] = beta  # bias


