
##################################################################################
# Builds a SavedModel which can be used for deployment on production environment #
# Updated by yuanpingzhou on 8/27/2018                                           #
##################################################################################

import os
import sys
import argparse

import tensorflow as tf

from model import OpenNsfwModel, InputType

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', "--target",
                        help="output directory",
                        default= '../data/model/nsfw/savedmodel',
                        )

    parser.add_argument("-v", "--export_version",
                        help="export model version",
                        default="1"
                        )

    parser.add_argument("-m", "--model_weights",
                        help="Path to trained model weights file",
                        default= '../data/model/nsfw/weight/open_nsfw-weights.npy',
                        )

    parser.add_argument("-t", "--input_type",
                        help="input type",
                        default=InputType.TENSOR.name.lower(),
                        #default= InputType.BASE64_JPEG.name.lower(),
                        choices=[InputType.TENSOR.name.lower(),InputType.BASE64_JPEG.name.lower()]
                        )


    args = parser.parse_args()

    model = OpenNsfwModel()

    export_base_path = args.target
    export_version = args.export_version

    export_path = os.path.join(export_base_path, export_version)

    with tf.Session() as sess:
        input_type = InputType[args.input_type.upper()]

        model.build(weights_path=args.model_weights,input_type= input_type)

        sess.run(tf.global_variables_initializer())

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # projected feature  signature
        nsfw_features_signature = (
            tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={'input_image': model.input},
                outputs={'nsfw_features': model.nsfw_features},
                #method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        # the final prediction signature
        prediction_signature = (
            tf.saved_model.signature_def_utils.predict_signature_def(
                inputs= {'input_image': model.input},
                outputs= {'probabilities': model.predictions},
                #method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_proba': prediction_signature,
                'projected_features': nsfw_features_signature,
            }
        )

        builder.save()
