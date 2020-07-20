# from src.io.config import parse_configuration
#
#
# parse_configuration(argv=['gpu', '4'])


from src.io.data import PointNetTfUtils
from src.train.aux_methods import batch_norm_template, max_pool2d, _variable_on_cpu, fully_connected

import tensorflow as tf


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  with tf.variable_scope(scope):
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w, num_in_channels, num_output_channels]
      kernel = _variable_on_cpu('weights', kernel_shape, tf.initializers.glorot_uniform())

      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel, [1, stride_h, stride_w, 1], padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels], tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_template(outputs, is_training, 'bn', [0, 1, 2], bn_decay)

      return activation_fn(outputs)


class Convolution2DLayer:

    def __init__(self, kernel_h, kernel_w, channels, filters, stride, padding,
                 batch_normalization, bn_decay,
                 activation_fn,
                 scope):
        self.output_channels = filters
        self.kernel_shape = [kernel_h, kernel_w, channels, filters]
        self.strides = [1, stride, stride, 1]
        self.scope = scope
        self.padding = padding
        self.batch_normalization = batch_normalization
        self.bn_decay = bn_decay
        self.activation_fn = activation_fn

    def run(self, inputs, is_training):
        with tf.variable_scope(self.scope):
            self.kernel = _variable_on_cpu('weights', self.kernel_shape, tf.initializers.glorot_uniform())
            outputs = tf.nn.conv2d(inputs, self.kernel, self.strides, padding=self.padding)

            self.biases = _variable_on_cpu('biases', [self.output_channels], tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(outputs, self.biases)

            if self.batch_normalization:
                outputs = batch_norm_template(outputs, is_training, 'batch_norm', [0, 1, 2], self.bn_decay)

            if self.activation_fn:
                outputs = self.activation_fn(outputs)

            return outputs


class FullyConnectedLayer:

    def __init__(self, n_in, n_out, batch_normalization, bn_decay, activation_fn, scope):
        self.n_in = n_in
        self.n_out = n_out
        self.batch_normalization = batch_normalization
        self.bn_decay = bn_decay
        self.activation_fn = activation_fn
        self.scope = scope

    def run(self, inputs, is_training):
        with tf.variable_scope(self.scope):

            self.weights = _variable_on_cpu('weights',
                                            shape=[self.n_in, self.n_out],
                                            initializer=tf.initializers.glorot_uniform())

            outputs = tf.matmul(inputs, self.weights)
            self.biases = _variable_on_cpu('biases', [self.n_out], tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(outputs, self.biases)

            if self.batch_normalization:
                outputs = batch_norm_template(outputs, is_training, 'batch_norm', [0, ], self.bn_decay)

            if self.activation_fn is not None:
                outputs = self.activation_fn(outputs)

            return outputs


class PointNetModel:

    def __init__(self, pn_utils):
        self.pn_utils = pn_utils

    def _define_placeholders_graph(self, pn_utils):
        with tf.name_scope('PointNetModel.placeholders'):
            self.input_pl, self.labels_pl, self.is_training_pl = self.pn_utils.get_placeholders()

    def _get_prepared_input(self):
        return tf.expand_dims(self.input_pl, -1)

    def _get_batch_size(self):
        # return self.input_pl.get_shape()[0].value
        return self.pn_utils.batch_size

    def _get_number_of_points(self):
        # return self.input_pl.get_shape()[1].value
        return self.pn_utils.num_points

    def _get_number_of_channels(self):
        # return self._get_prepared_input().get_shape()[-1].value
        return self.pn_utils.input_channels

    def _define_classification_graph(self):

        conv2d_1 = Convolution2DLayer(1, 3, self._get_number_of_channels(), 64, 1, 'VALID',
                                      True, 0.9, None, 'conv1')
        net = conv2d_1.run(self._get_prepared_input(), self.is_training_pl)

        assert net.get_shape()[-1] == 64, 'dezavou; oh no unexpected sizes'

        conv2d_2 = Convolution2DLayer(1, 1, 64, 64, 1, 'VALID',
                                      True, 0.9, None, 'conv2')

        net = conv2d_2.run(net, self.is_training_pl)

        assert net.get_shape()[-1] == 64, 'dezavou; oh no unexpected sizes'

        conv2d_3 = Convolution2DLayer(1, 1, 64, 64, 1, 'VALID',
                                      True, 0.9, None, 'conv3')

        net = conv2d_3.run(net, self.is_training_pl)

        assert net.get_shape()[-1] == 64, 'dezavou; oh no unexpected sizes'

        conv2d_4 = Convolution2DLayer(1, 1, 64, 128, 1, 'VALID',
                                      True, 0.9, None, 'conv4')

        net = conv2d_4.run(net, self.is_training_pl)

        assert net.get_shape()[-1] == 128

        conv2d_5 = Convolution2DLayer(1, 1, 128, 1024, 1, 'VALID',
                                      True, 0.9, None, 'conv5')

        net = conv2d_5.run(net, self.is_training_pl)

        assert net.get_shape()[-1] == 1024

        # Symmetric function: max pooling
        net = max_pool2d(net, [self.pn_utils.num_points, 1], padding='VALID', scope='maxpool')

        # MLP on global point cloud vector
        net = tf.reshape(net, [self.pn_utils.batch_size, -1])

        fc1 = FullyConnectedLayer(net.get_shape()[-1], 512, True, 0.9, None, 'fc1')

        net = fc1.run(net, self.is_training_pl)


        net = fully_connected(net, 256, bn=True, is_training=is_training,
                              scope='fc2', bn_decay=bn_decay)

        with tf.variable_scope('dp1'):
            net = tf.cond(is_training, lambda: tf.nn.dropout(net, 0.7, None), lambda: net)

        net = fully_connected(net, 40, activation_fn=None, scope='fc3')

        return net, end_points

class TrainingParameters:

    def __init__(self, bn_decay=0.9):
        self.bn_decay = tf.Variable(bn_decay)


class TensorSummarizer:

    def __init__(self, dir):
        self.dir = dir

    def update_summary(self, name, tensor, type='scalar'):
        tf.summary.scalar(name, tensor)


class Trainer:

    def __init__(self, device, training_params, summarizer,
                 batch_size, num_points, input_channels, output_dim):
        self.device = device

        self.pn_utils = PointNetTfUtils(batch_size, num_points, input_channels, output_dim)
        self.input_pl, self.labels_pl, self.is_training_pl = self.pn_utils.get_placeholders()

        self.training_params = training_params
        self.summarizer = summarizer

        self.summarizer.update_summary('bn_decay', self.training_params.bn_decay)

        self.batch = tf.Variable(0)

    def run(self, model):
        pass


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):

            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        # merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'tests'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        # sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

