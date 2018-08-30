#!/usr/bin/env python
import sys
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader

import numpy as np
import glob


IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', "--input_dir", default= 'test',
                        help="Path to the input image. Only jpeg images are supported.")

    parser.add_argument("-m", "--model_weights", default= 'data/open_nsfw-weights.npy',
                        help="Path to trained model weights file")

    parser.add_argument("-l", "--image_loader",default=IMAGE_LOADER_YAHOO,
                        help="image loading mechanism",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    parser.add_argument("-t", "--input_type",default=InputType.TENSOR.name.lower(),
                        help="input type",
                        choices=[InputType.TENSOR.name.lower(),
                                 InputType.BASE64_JPEG.name.lower()])

    args = parser.parse_args()

    image_files = glob.glob('%s/*.jpg' % args.input_dir)

    model = OpenNsfwModel()

    predictions = []

    with tf.Session() as sess:

        input_type = InputType[args.input_type.upper()]
        model.build(weights_path=args.model_weights, input_type=input_type)

        fn_load_image = None

        if input_type == InputType.TENSOR:
            if args.image_loader == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(sess)
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64
            fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

        sess.run(tf.global_variables_initializer())

        for i, img_file in enumerate(image_files):
            image = fn_load_image(img_file)
            predictions.append(sess.run(model.predictions,feed_dict={model.input: image})[0][1])

    columns = 4
    rows = int(len(image_files) / 4) + int((len(image_files) % 4) > 0)
    print((rows, columns))
    fig = plt.figure(figsize=(12, 16), dpi=100)
    fig.subplots_adjust(hspace=0.6, wspace=0.1)
    images = []
    for i, img_file in enumerate(image_files):
        img = plt.imread(img_file)
        fig.add_subplot(rows, columns, i + 1, xlabel='%.4f' % predictions[i])
        plt.imshow(img)
    plt.savefig('demo.jpg')

if __name__ == "__main__":
    main(sys.argv)
