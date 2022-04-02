import logging
import os

import numpy as np

IMAGE_WIDTH = 28
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH  # e.g. 28x28 for image and 1 byte for label


def read_data_from_single_file(fname, count=None):
    data = []
    stop_flag = False
    idx = 0

    with open(fname, 'rb') as f:
        while not stop_flag and (count is None or idx < count):
            img_data = f.read(IMAGE_SIZE)
            if len(img_data) == IMAGE_SIZE:
                data.append(byte_arr_to_int_arr(img_data))
                idx += 1
            else:
                stop_flag = True
    logging.info(f"{fname} loaded. count: {idx} imgs")
    return data


def byte_arr_to_int_arr(bytearr):
    return list(bytearr)


def load_data(data_dir, count_per_file=None):
    data = []

    for filename in os.listdir(data_dir):
        logging.info(f"{filename} loaded.")
        imagePath = data_dir + "/" + filename
        data += read_data_from_single_file(imagePath, count_per_file)

    if data:
        return extract_images_labels(data)
    else:
        logging.warning("No data loaded")
        return None


def extract_images_labels(data):
    img_data = [np.reshape(f_data, (IMAGE_WIDTH, IMAGE_WIDTH, 1)) for f_data in data]
    return img_data

