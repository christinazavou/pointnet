from src.train.aux_methods import batch_norm_template

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import unittest


class TestAuxMethods(unittest.TestCase):

    def test_batch_norm_template(self):
        # normally the activations are parameters .. but to test this layer we need placeholder ..
        activations = tf.placeholder(tf.float32, shape=(8, 10, 1, 6))
        # activations = tf.variable(tf.float32, shape=(8, 10, 1, 6))
        is_training = tf.placeholder(tf.bool, shape=())
        normalized_activations = batch_norm_template(activations, is_training, 'bn', [0,1,2], 0.9)
        print("\n"+normalized_activations+"\n")


if __name__ == '__main__':
    unittest.main()
