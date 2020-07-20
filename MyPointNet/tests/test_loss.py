
from src.train.loss import ClassificationLossCalculator

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from unittest import TestCase


class TestClassificationLossCalculator(TestCase):

    def setUp(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.config.log_device_placement = False

    def test_calculate(self):

        clc = ClassificationLossCalculator()

        def v1():
            logits = tf.get_variable('logits', dtype=tf.float32, shape=(4, 5))
            labels = tf.placeholder(tf.int32, shape=(4))

            with tf.Session(config=self.config) as sess:
                feed_dict = {labels: [3, 2, 1, 0]}
                logits_op = tf.assign(logits, [
                    [0.2, 0.3, 0.5, 0.6, 0.4],
                    [0.1, 0.1, 0.5, 0.4, 0.4],
                    [0.9, 0.01, 0.2, 0.3, 0.01],
                    [0.2, 0.6, 0.4, 0.3, 0.3]
                ])
                _, loss = sess.run([logits_op, clc.calculate(logits, labels, 'val')], feed_dict=feed_dict)
                print("\nloss: {}\n".format(loss))
            return loss

        def v2():
            logits = tf.Variable([
                [0.2, 0.3, 0.5, 0.6, 0.4],
                [0.1, 0.1, 0.5, 0.4, 0.4],
                [0.9, 0.01, 0.2, 0.3, 0.01],
                [0.2, 0.6, 0.4, 0.3, 0.3]
            ], dtype=tf.float32, shape=(4, 5))
            labels = tf.placeholder(tf.int32, shape=(4))

            with tf.Session(config=self.config) as sess:
                feed_dict = {labels: [3, 2, 1, 0]}
                sess.run(tf.global_variables_initializer())
                loss = sess.run(clc.calculate(logits, labels, 'val'), feed_dict=feed_dict)
                print("\nloss: {}\n".format(loss))
            return loss

        def v3():
            logits = tf.placeholder(tf.float32, shape=(4, 5))
            labels = tf.placeholder(tf.int32, shape=(4))

            with tf.Session(config=self.config) as sess:
                feed_dict = {
                    logits:[
                        [0.2, 0.3, 0.5, 0.6, 0.4],
                        [0.1, 0.1, 0.5, 0.4, 0.4],
                        [0.9, 0.01, 0.2, 0.3, 0.01],
                        [0.2, 0.6, 0.4, 0.3, 0.3]
                    ],
                    labels: [3, 2, 1, 0]
                }
                loss = sess.run(clc.calculate(logits, labels, 'val'), feed_dict=feed_dict)
                print("\nloss: {}\n".format(loss))
            return loss

        self.assertTrue(v1() == v2() == v3())
