
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class ClassificationLossCalculator:

    def __init__(self):
        self.loss_name = 'classification loss'

    def calculate(self, logits, labels, scope):
        """
        :param logits: UNSCALED logits of shape batch_size x categories
        :param labels: batch_size
        :param scope: train or tests or validation
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('{} {}'.format(scope, self.loss_name), loss)
        return loss

