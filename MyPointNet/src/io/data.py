import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class PointNetTfUtils:

    def __init__(self, batch_size, num_points, input_channels, output_dim):
        self.batch_size = batch_size
        self.num_points = num_points
        self.input_channels = input_channels
        self.output_dim = output_dim

    def get_placeholders(self):
        input_pl = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_points, self.input_channels))
        labels_pl = tf.placeholder(tf.int32, shape=(self.batch_size))
        is_training_pl = tf.placeholder(tf.bool, shape=())
        return input_pl, labels_pl, is_training_pl

