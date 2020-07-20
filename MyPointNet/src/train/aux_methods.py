import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


"""-----------------------------------------------------------------------------------------------------------
Note1:
Batch Norm needs to calculate the mean and variance of the whole batch.
These needs to be calculated during the training, but since they are not used to calculate the loss
by default our graph will not update them. Thus, we need to manually update them while training.
These (mean and variance) are needed for the inference.

Two variables, gamma and beta, are the ones responsible for rescaling and offsetting the activations 
within "BatchNormalization Layer".

(Theoretically the FullyConvolutional Layer after the "BatchNormalization Layer" should not use biases)

--------------------------------------------------------------------------------------------------------------
Note2:
Batch Normalization is a solution to the unstable gradients within DEEP neural networks. 
BN method introduces an additional layer to the NN that performs operations on the inputs from the previous layer.

Usually when we train or tests we use a specific batch size of input data. We need to calculate the statistics of the training data and use those in the tests data as well. Normally you want to have training data with similar statistics to the tests data, otherwise your NN won't perform well.

Normalization is typically done in the Input Layer..but it makes sense to do it also in internal layers to limit
the covariance shift that occurs from activations...thus BN method!!

#Note3:
Batch normalization does:
1. standardization of input batch:
 x_hat = (x_i - mean) / sqrt(variance + epsilon)
 where epsilon is usually 0.00005 and is to ensure numerical stability
and then
2. rescaling and offsetting of the standardize value:
gamma*x_hat + beta

(doing rescale +offset is basically shifting again the covariance....but it's found out to boost performance!)

#Note4:
While μ and σ² parameters are estimated from the input data, γ and β are trainable.
Thus, they can be leveraged by the back-propagation algorithm to optimize the network.

# original paper:
https://arxiv.org/abs/1502.03167

# Note5:
since we want to have one mean and one variance in the inference state, found out from the training data,
we use exponential moving average over the batches!

# Note6:
Exponential Moving Average needs a decay term (close to 1).
Using Tensorflow implementation, the moving average is calculated as:
shadow_variable -= (1 - decay) * (shadow_variable - variable)
and the apply() method is the one to update our moving average
-------------------------------------------------------------------------------------------------------------"""

class BatchNormalizationLayer:

    def __init__(self, scope, input_layer_name, activations_dim, axes_is_batch, ema_decay=0.9):
        # note: since batch normalization layer has as "input" or "features" the activations of the previous
        # layer, i use "activations_dim"
        self.scope = scope
        self.input_layer_name = input_layer_name
        self.activations_dim = activations_dim
        self.axes = [0] if axes_is_batch else [0, 1, 2]
        self.ema_decay = ema_decay

        with tf.variable_scope(self.scope):

            self.beta = tf.Variable(tf.constant(0.0, shape=[self.activations_dim]), name='beta', trainable=True)
            self.gamma = tf.Variable(tf.constant(1.0, shape=[self.activations_dim]), name='gamma', trainable=True)

            self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)

    # Update moving average and return current batch's avg and var.
    @staticmethod
    def mean_var_with_update(batch_mean, batch_var, ema_apply_op):
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    def run_on_batch(self, inputs, is_training):
        with tf.variable_scope(self.scope):
            batch_mean, batch_var = tf.nn.moments(inputs, self.axes, name='moments')

            # Operator that maintains moving averages of variables.
            ema_apply_op = tf.cond(is_training,
                                   true_fn=lambda: self.ema.apply([batch_mean, batch_var]),
                                   false_fn=lambda: tf.no_op())

            # ema.average returns the Variable holding the average of var.
            mean, var = tf.cond(is_training,
                                true_fn=self.mean_var_with_update(batch_mean, batch_var, ema_apply_op),
                                false_fn=lambda: (self.ema.average(batch_mean), self.ema.average(batch_var)))
            normed = tf.nn.batch_normalization(inputs, mean, var, self.beta, self.gamma, 1e-3)
        return normed


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    return tf.get_variable(name, shape, initializer=initializer, dtype=dtype)


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_ema_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_ema_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value

        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')

        decay = bn_ema_decay if bn_ema_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               true_fn=lambda: ema.apply([batch_mean, batch_var]),
                               false_fn=lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            true_fn=mean_var_with_update,
                            false_fn=lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D max pooling.

    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_input_units, num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weight_decay)
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_template(outputs, is_training, 'bn', [0, ], bn_decay)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    # initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.initializers.glorot_uniform()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

