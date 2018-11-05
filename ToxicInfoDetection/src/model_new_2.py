import math
import numpy as np
import tensorflow as tf
from keras.layers.convolutional import UpSampling2D
from enum import Enum, unique

@unique
class InputType(Enum):
    TENSOR = 1
    BASE64_JPEG = 2

class OpenNSFW:
    """Tensorflow implementation of Yahoo's Open NSFW Model

    Original implementation:
    https://github.com/yahoo/open_nsfw

    Weights have been converted using caffe-tensorflow:
    https://github.com/ethereon/caffe-tensorflow
    
    @Attention params
    :param p: the number of pre-processing Residual Units before splitting into trunk branch and mask branch
    :param t: the number of Residual Units in trunk branch
    :param r: the number of Residual Units between adjacent pooling layer in the mask branch
    """
    def __init__(self, weights_path = 'open_nsfw-weights.npy', num_classes = 2, p= 1, t= 2, r= 1):
        self.weights = np.load(weights_path, encoding="latin1").item()
        self.bn_epsilon = 1e-5  # Default used by Caffe
        self.num_classes = num_classes
        self.p = p
        self.t = t
        self.r = r

    def build(self, input_type=InputType.TENSOR):

        #self.weights = np.load(weights_path, encoding="latin1").item()
        self.input_tensor = None
        self.training = tf.placeholder(tf.bool, name= 'training_mode')
        self.learning_rate = tf.placeholder(tf.float32, name= 'learning_rate')

        if input_type == InputType.TENSOR:
            self.input = tf.placeholder(tf.float32,
                                        #shape=[None, 224, 224, 3],
                                        shape= [None, 256, 256, 3],
                                        name="input")
            self.input_tensor = self.input
        elif input_type == InputType.BASE64_JPEG:
            from image_utils import load_base64_tensor

            self.input = tf.placeholder(tf.string, shape=(None,), name="input")
            self.input_tensor = load_base64_tensor(self.input)
        else:
            raise ValueError("invalid input type '{}'".format(input_type))

        x = self.input_tensor
        self.y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(self.y, self.num_classes)

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'CONSTANT')
        x = self.__conv2d("conv_1", x, filter_depth=64,
                          kernel_size=7, stride=2, padding='valid')

        x = self.__batch_norm("bn_1", x)
        x = tf.nn.relu(x)

        x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same')

        ## attenion block
        x = self.__attention_block(stage= 0, block= 0, inputs= x,
                                   filter_depths= [32, 32, 128], kernel_size= 3, stride= 1)

        x = self.__conv_block(stage=0, block=0, inputs=x,
                              filter_depths=[32, 32, 128],
                              kernel_size=3, stride=1, scope= 'res0')

        x = self.__identity_block(stage=0, block=1, inputs=x,
                                  filter_depths=[32, 32, 128], kernel_size=3)
        x = self.__identity_block(stage=0, block=2, inputs=x,
                                  filter_depths=[32, 32, 128], kernel_size=3)

        ## attenion block
        x = self.__attention_block(stage= 1, block= 0, inputs= x,
                                   filter_depths= [64, 64, 256], kernel_size= 3, stride= 2)

        x = self.__conv_block(stage=1, block=0, inputs=x,
                              filter_depths=[64, 64, 256],
                              kernel_size=3, stride=2, scope= 'res1')
        x = self.__identity_block(stage=1, block=1, inputs=x,
                                  filter_depths=[64, 64, 256], kernel_size=3)
        x = self.__identity_block(stage=1, block=2, inputs=x,
                                  filter_depths=[64, 64, 256], kernel_size=3)
        x = self.__identity_block(stage=1, block=3, inputs=x,
                                  filter_depths=[64, 64, 256], kernel_size=3)

        ## attenion block
        x = self.__attention_block(stage= 2, block= 0, inputs= x,
                                   filter_depths= [128, 128, 512], kernel_size= 3, stride= 2)

        x = self.__conv_block(stage=2, block=0, inputs=x,
                              filter_depths=[128, 128, 512],
                              kernel_size=3, stride=2, scope= 'res2')
        x = self.__identity_block(stage=2, block=1, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        x = self.__identity_block(stage=2, block=2, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        x = self.__identity_block(stage=2, block=3, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        x = self.__identity_block(stage=2, block=4, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)
        x = self.__identity_block(stage=2, block=5, inputs=x,
                                  filter_depths=[128, 128, 512], kernel_size=3)

        x = self.__conv_block(stage=3, block=0, inputs=x,
                              filter_depths=[256, 256, 1024], kernel_size=3,
                              stride=2, scope= 'res3')
        x = self.__identity_block(stage=3, block=1, inputs=x,
                                  filter_depths=[256, 256, 1024],
                                  kernel_size=3)
        x = self.__identity_block(stage=3, block=2, inputs=x,
                                  filter_depths=[256, 256, 1024],
                                  kernel_size=3)

        #x = tf.layers.average_pooling2d(x, pool_size=7, strides=1,
        #                                padding="valid", name="pool")

        x = tf.reshape(x, shape= (-1, 1024), name= 'nsfw_features')

        # output
        self.logits = self.__fully_connected(name="fc_nsfw", inputs= x, num_outputs= self.num_classes)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= one_hot_y, logits= self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate).minimize(self.loss)

        ## accuracy
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), self.y)
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        ## recall
        toxic_predicts = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), tf.cast(2, tf.int32))
        toxic_labels = tf.equal(self.y, tf.cast(2, tf.int32))
        self.toxic_recall = tf.metrics.recall(toxic_labels, toxic_predicts)
        self.toxic_label_sum = tf.reduce_sum(tf.cast(toxic_labels, tf.int32))

        ## precision
        self.toxic_precision = tf.metrics.precision(toxic_labels, toxic_predicts)
        self.toxic_predict_sum = tf.reduce_sum(tf.cast(toxic_predicts, tf.int32))

    """Get weights for layer with given name
    """
    def __get_weights(self, layer_name, field_name):
        if not layer_name in self.weights:
            raise ValueError("No weights for layer named '{}' found"
                             .format(layer_name))

        w = self.weights[layer_name]
        if not field_name in w:
            raise (ValueError("No entry for field '{}' in layer named '{}'"
                              .format(field_name, layer_name)))

        return w[field_name]

    """Layer creation and weight initialization
    """
    def __fully_connected(self, name, inputs, num_outputs):
        return tf.layers.dense(
            inputs=inputs, units=num_outputs, name=name,
            kernel_initializer=tf.constant_initializer(
                self.__get_weights(name, "weights"), dtype=tf.float32),
            bias_initializer=tf.constant_initializer(
                self.__get_weights(name, "biases"), dtype=tf.float32))

    def __conv2d(self, name, inputs, filter_depth, kernel_size, stride=1,
                 padding="same", trainable=False):

        if padding.lower() == 'same' and kernel_size > 1:
            if kernel_size > 1:
                oh = inputs.get_shape().as_list()[1]
                h = inputs.get_shape().as_list()[1]

                p = int(math.floor(((oh - 1) * stride + kernel_size - h)//2))

                inputs = tf.pad(inputs,
                                [[0, 0], [p, p], [p, p], [0, 0]],
                                'CONSTANT')
            else:
                raise Exception('unsupported kernel size for padding: "{}"'
                                .format(kernel_size))

        # return tf.layers.conv2d(
        #     inputs, filter_depth,
        #     kernel_size=(kernel_size, kernel_size),
        #     strides=(stride, stride), padding='valid',
        #     activation=None, trainable=trainable, name=name,
        #     kernel_initializer=tf.constant_initializer(
        #         self.__get_weights(name, "weights"), dtype=tf.float32),
        #     bias_initializer=tf.constant_initializer(
        #         self.__get_weights(name, "biases"), dtype=tf.float32))

        ## no initialization
        return tf.layers.conv2d(
            inputs, filter_depth,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride), padding='valid',
            activation=None, trainable=trainable, name=name)

    # def __batch_norm(self, name, inputs):
    #     return tf.layers.batch_normalization(
    #         inputs, training= self.training, epsilon=self.bn_epsilon,
    #         gamma_initializer=tf.constant_initializer(
    #             self.__get_weights(name, "scale"), dtype=tf.float32),
    #         beta_initializer=tf.constant_initializer(
    #             self.__get_weights(name, "offset"), dtype=tf.float32),
    #         moving_mean_initializer=tf.constant_initializer(
    #             self.__get_weights(name, "mean"), dtype=tf.float32),
    #         moving_variance_initializer=tf.constant_initializer(
    #             self.__get_weights(name, "variance"), dtype=tf.float32),
    #         name=name)

    def __batch_norm(self, name, inputs):
        return tf.layers.batch_normalization(
            inputs, training= self.training, epsilon=self.bn_epsilon,
            name=name)

    '''
    Attention blocks
    '''
    def __attention_block(self, stage, block, inputs, filter_depths, kernel_size= 3, stride= 2):
        ''''''

        ## first residual block
        for i in range(self.p):
            inputs = self.__conv_block(stage, block, inputs, filter_depths, kernel_size, stride,
                                      scope="attention_first_residual_block_num_blocks_{}".format(i))

        ## trunk branch
        output_trunk = inputs
        for i in range(self.t):
            output_trunk = self.__conv_block(stage, block, output_trunk, filter_depths,
                                             scope="attention_trunk_branch_num_blocks_{}".format(i))

        ## softmax branch

        ## down sampling 0
        filter_ = [1, 2, 2, 1]
        output_soft_mask = tf.nn.max_pool(inputs, ksize=filter_, strides=filter_, padding='SAME', name= 'attention_softmax_branch_down_sampling_1')

        for i in range(self.r):
            output_soft_mask = self.__conv_block(stage, block, output_soft_mask, filter_depths,
                                                          scope="attention_softmax_branch_down_sampling_1_num_blocks_{}".format(i))
        ### upsampling 0
        #for i in range(self.r):
        #    output_soft_mask = self.__conv_block(stage, block, output_soft_mask, filter_depths,
        #                                                  scope="attention_softmax_branch_up_sampling_0_num_blocks_{}".format(i))
        ## interpolation
        #output_soft_mask = UpSampling2D([2, 2], name= 'attention_softmax_branch_up_sampling_0')(output_soft_mask)

        ## skip connection
        output_skip_connection = self.__conv_block(stage, block, output_soft_mask, filter_depths,
                                                   scope= 'attention_softmax_branch_skip_connection')
        ## down sampling 1
        filter_ = [1, 2, 2, 1]
        output_soft_mask = tf.nn.max_pool(output_soft_mask, ksize=filter_, strides=filter_, padding='SAME', name= 'attention_softmax_branch_down_sampling_2')

        for i in range(self.r):
            output_soft_mask = self.__conv_block(stage, block, output_soft_mask, filter_depths,
                                                          scope="attention_softmax_branch_down_sampling_2_num_blocks_{}".format(i))
        ## upsampling 1
        for i in range(self.r):
            output_soft_mask = self.__conv_block(stage, block, output_soft_mask, filter_depths,
                                                          scope="attention_softmax_branch_up_sampling_1_num_blocks_{}".format(i))
        # interpolation
        output_soft_mask = UpSampling2D([4, 4], name= 'attention_softmax_branch_up_sampling_1')(output_soft_mask)

        output_soft_mask = tf.add(output_soft_mask, output_skip_connection, name= 'attention_softmax_branch_add')

        ##  upsampling 2
        for i in range(self.r):
            output_soft_mask = self.__conv_block(stage, block, output_soft_mask, filter_depths,
                                                          scope="attention_softmax_branch_up_sampling_2_num_blocks_{}".format(i))

        # interpolation
        output_soft_mask = UpSampling2D([4, 4], name= 'attention_softmax_branch_up_sampling_2')(output_soft_mask)

        ## output
        output_soft_mask = tf.layers.conv2d(output_soft_mask, filters= filter_depths[-1], kernel_size=1)#, name= 'attention_softmax_branch_output_conv_1_1')
        output_soft_mask = tf.layers.conv2d(output_soft_mask, filters= filter_depths[-1], kernel_size=1)#, name= 'attention_softmax_branch_output_conv_2_2')

        # sigmoid
        output_soft_mask = tf.nn.sigmoid(output_soft_mask, name= 'attention_softmax_branch_output_sigmoid')

        output = tf.multiply(1 + output_soft_mask, output_trunk, name= 'attention_multiply')

        ## last residual
        for i in range(self.p):
            output = self.__conv_block(stage, block, output, filter_depths,
                                       scope="attention_last_residual_block_num_blocks_{}".format(i))

        return output

    """ResNet blocks
    """
    def __conv_block(self, stage, block, inputs, filter_depths,
                     kernel_size=3, stride=2, scope= ''):
        filter_depth1, filter_depth2, filter_depth3 = filter_depths

        conv_name_base = "{}_conv_stage{}_block{}_branch".format(scope, stage, block)
        bn_name_base = "{}_bn_stage{}_block{}_branch".format(scope, stage, block)
        shortcut_name_post = "{}_stage{}_block{}_proj_shortcut" \
                             .format(scope, stage, block)

        shortcut = self.__conv2d(
            name="conv{}".format(shortcut_name_post), stride=stride,
            inputs=inputs, filter_depth=filter_depth3, kernel_size=1,
            padding="same"
        )

        shortcut = self.__batch_norm("bn{}".format(shortcut_name_post),
                                     shortcut)

        x = self.__conv2d(
            name="{}2a".format(conv_name_base),
            inputs=inputs, filter_depth=filter_depth1, kernel_size=1,
            stride=stride, padding="same",
        )
        x = self.__batch_norm("{}2a".format(bn_name_base), x)
        x = tf.nn.relu(x)

        x = self.__conv2d(
            name="{}2b".format(conv_name_base),
            inputs=x, filter_depth=filter_depth2, kernel_size=kernel_size,
            padding="same", stride=1
        )
        x = self.__batch_norm("{}2b".format(bn_name_base), x)
        x = tf.nn.relu(x)

        x = self.__conv2d(
            name="{}2c".format(conv_name_base),
            inputs=x, filter_depth=filter_depth3, kernel_size=1,
            padding="same", stride=1
        )
        x = self.__batch_norm("{}2c".format(bn_name_base), x)

        x = tf.add(x, shortcut)

        return tf.nn.relu(x)

    def __identity_block(self, stage, block, inputs,
                         filter_depths, kernel_size):
        filter_depth1, filter_depth2, filter_depth3 = filter_depths
        conv_name_base = "conv_stage{}_block{}_branch".format(stage, block)
        bn_name_base = "bn_stage{}_block{}_branch".format(stage, block)

        x = self.__conv2d(
            name="{}2a".format(conv_name_base),
            inputs=inputs, filter_depth=filter_depth1, kernel_size=1,
            stride=1, padding="same",
        )

        x = self.__batch_norm("{}2a".format(bn_name_base), x)
        x = tf.nn.relu(x)

        x = self.__conv2d(
            name="{}2b".format(conv_name_base),
            inputs=x, filter_depth=filter_depth2, kernel_size=kernel_size,
            padding="same", stride=1
        )
        x = self.__batch_norm("{}2b".format(bn_name_base), x)
        x = tf.nn.relu(x)

        x = self.__conv2d(
            name="{}2c".format(conv_name_base),
            inputs=x, filter_depth=filter_depth3, kernel_size=1,
            padding="same", stride=1
        )
        x = self.__batch_norm("{}2c".format(bn_name_base), x)

        x = tf.add(x, inputs)

        return tf.nn.relu(x)
