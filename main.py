import argparse

import caffe
import keras.models

import caffe_model
from keras_weights_to_caffe_model import keras_weights_to_caffe_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str,
                        help="Path to the .h5 Keras model with weights", required=True)
    return parser.parse_args()

def main():
    argument = parse_args()
    model_name = argument.model
    #caffe_model.create_model()
    c_model = caffe.Net("model_test.prototxt", caffe.TRAIN)
    model = keras.models.load_model(model_name)
    keras_weights_to_caffe_model(model,c_model)

if __name__ == '__main__':
    main()

