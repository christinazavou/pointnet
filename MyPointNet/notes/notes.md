Questions

- why input_image shape is (8,1024,3,1) i.e. each channel has 1 value? is it the intensity (i.e. is 3 for X,Y,Z of the point and 1 for a greay scale image) ?
- in conv2D the num_output_channels is basically the amount of filters ?
- ~~why the _variable_on_cpu call in variable_with_weight_decay?~~ 
    - check https://stackoverflow.com/questions/34428850/variables-on-cpu-training-gradients-on-gpu
    - check https://jhui.github.io/2017/03/07/TensorFlow-GPU/
    - basically it depends on hardware configuration...and it can be more efficient to place variables on the CPU
- what are second-order optimization methods that attempt to model the curvature of the cost surface?
- ~~using pointnet_cls_basic...i see in that predictions (pred_val train.py line 205) are of shape batchSize,40 ... so i guess we have 40 categories~~...yes because also labels are numbers from 0 to 39...which are these categories ?
- why do we take the first points as samples of the point cloud ? (line 184 in train.py) is it because anyway points in the cloud are unordered?
- for example in function 'fully_connected' there is a parameter weiht_decay and a parameter bn_decay .. what exactly are they?

Interesting

- function _variable_with_weight_decay makes a variable and if this variable is for example a weight that we want to be included in the regularization (e.g. L2) term then it also creates a variable for the weight decay term (regarding this weight variable) and adds it to the loss collection. this is a collection to keep all our loss terms! i.e.:
    ```python
      if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    ```

- There are two types of parallelism:
    - Model parallelism 
        - Different GPUs run different part of the code. Batches of data pass through all GPUs. e.g.:
    ```python
    import tensorflow as tf
    
    c = []
    a = tf.get_variable(f"a", [2, 2], initializer=tf.random_uniform_initializer(-1, 1))
    b = tf.get_variable(f"b", [2, 2], initializer=tf.random_uniform_initializer(-1, 1))
    
    with tf.device('/gpu:0'):
        c.append(tf.matmul(a, b))
    
    with tf.device('/gpu:1'):
        c.append(a + b)
    
    with tf.device('/cpu:0'):
        sum = tf.add_n(c)
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    print(sess.run(sum))
    ```
    
    - Data parallelism 
        - Multiple GPUs run the same TensorFlow code. Each GPU is feed with different batch of data.
        - If a host have multiple GPUs with the same memory and computation capacity, it will be simpler to scale with data parallelism. e.g.:
      ```python
      import tensorflow as tf
      
      c = []
      a = tf.get_variable(f"a", [2, 2, 3], initializer=tf.random_uniform_initializer(-1, 1))
      b = tf.get_variable(f"b", [2, 3, 2], initializer=tf.random_uniform_initializer(-1, 1))
      
      # Multiple towers
      for i, d in enumerate(['/gpu:0', '/gpu:1']):
          with tf.device(d):
              c.append(tf.matmul(a[i], b[i]))
          # Tower i is responsible for batch data i.
          with tf.device('/cpu:0'):
              sum = tf.add_n(c)
      sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
      init = tf.global_variables_initializer()
      sess.run(init)
      print(sess.run(sum))
      ```

- the conv2d has as arguments strides of shape 1,2,or 4.
    - if 4 then it is stride for [N,H,W,C] i.e. stride for batch, height, width and channel. normally you set N to 1 (you don't want to skip any batch otherwise you wouldnt include those data in training) and C to 1 (you don't want to skip data - channel information).
  
- _**When you have a large dataset it's important to optimize well and not as important to regularize well**_. Batch normalization firstly an optimization technique but it also provides regularization. 
    - It keeps the back propagated gradient from getting too big or too small, by rescaling and recentering. This is useful also for when adding random noise (_**e.g. dropout or bias ?!**_)
    - Batch norm is similar to dropout (can be seen as multiplying each weight with 0 or 1) in the sense that it multiplies each hidden unit by a random value at each step of training. In this case, the random value is the standard deviation of all the hidden units in the minibatch (because different examples are randomly chosen for inclusion in the minibatch at each step, the standard deviation randomly fluctuates). It also subtracts a random value (the mean of the minibatch) from each hidden unit at each step. Both of these sources of noise mean that every layer has to learn to be robust to a lot of variation in its input, just like with dropout.
    - benefits:
        - train faster (even if training an epoch takes longer ...it will eventually converge faster)
        - allows higher learning rates (so converges faster)
        - weight initialization can be difficult but Batch Normalization helps reduce the sensitivity to the initial starting weights.
        - more activation functions viable

- Exponentially weighted averages:
    - V_t = βV_(t-1) + (1-β)θ_t
  where V_t is the expected temperature today and θ_t is the real temperature today
  so V_t is approximately the average temperature over the previous 1/(1-β) days
  so e.g. V_100 = 0.9V_99 + 0.1θ_98 = 0.9V_99 + 0.1(0.9θ_98) + 0.1(0.1θ_97) = ... = 0.9V_99 + 0.1(0.9)^2θ_97 0.1*0.1θ_96+ 0.1(0.1)*0.9θ_96+0.1*0.1*0.1*θ_95 .... so the coefficients follow an exponentailly decaying function  
  Calculating this means just keep one number in memory and keep overwritting it (start with V_0)..so it's efficient!! (in oppose to real average of last days were you keep all days in memory)  
  Note that V_1 is only using V_0 so it's not as good estimation as later estimations like V_10 ... so we need to do: bias correction !

    - Bias correction:
        - multiplying your estimation by 1/(1-β^t) .. so in initial timesteps (1-β^t) is near zero and as t grows it approaches to one.
    - Basically in batch normalization we need mean (activations) of batch and variance (activations) of batch .. and to get them we use exponentially weighted averages!!
  
    - Batch normalization in tensorflow:
        - ![](batchnorm1.png)
        - self.x_norm, moving_update = tf.cond(is_training, true_fn=training_fn, false_fn=testing_fn)


Notes

- TensorFlow variables:
    - used to share and persist some stats that are manipulated by our program. That is, when you define a variable, TensorFlow adds a tf.Operation to your graph. Then, this operation will store a writable tensor value that persists between tf.Session.run calls. 
    - you can update the value of a variable through each run, while you cannot update tensor (e.g a tensor created by tf.constant()) through multiple runs in a session.
    - To define variables we use the command tf.Variable(). 
    - To be able to use variables in a computation graph it is necessary to initialize them before running the graph in a session. This is done by running tf.global_variables_initializer().

- update = tf.assign(v, v+1)

- Placeholders can be seen as “holes” in your model, so when we initialize the session we are obligated to pass an argument with the data, otherwise we would get an error.

- example snippet:
    ```python
    graph5 = tf.Graph()
    with graph5.as_default():
        a = tf.constant([5])
        b = tf.constant([2])
        c = tf.add(a,b)
        d = tf.subtract(a,b)
    with tf.Session(graph = graph5) as sess:
        result = sess.run(c)
        print ('c =: %s' % result)
        result = sess.run(d)
        print ('d =: %s' % result)
    ```

- To handle multiple devices configuration, set allow_soft_placement to True. It places the operation into an alternative device automatically. Otherwise, the operation will throw an exception if the device does not exist.
example snippet:
    ```python
    sess = tf.Session(config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True))
    ```
  
- TensorFlow can grow its memory gradually by (if desired):
    ```python
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config, ...)
    ```
  Or to specify that we want say 40% of the total GPUs memory.
    ```python
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.Session(config=config, ...)
    ```

- standardization: transform your data set to have zero mean and unit variance

- normalization: important preprocessing when we deal with parameters of different units and scales. e.g. if we want to use Euclidean distance we need to normalize our points (e.g. into meters) to compare them..

- _**Batch normalization is a technique to provide any layer in a NN with inputs that are zero mean / unit variance**_

- dropout is a regularization technique

- we can achieve batch normalization with 
    - tf.layers.batch_normalization
        a trainable layer that learns a function of two parameters (gamma and beta) so that we don't need to standardize its input data :) 
        _**so basically is a neural layer with weights and biases that need to be learned..and we learn them with exponential moving average!? (as i understood from https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)**_
        ![](batchnorm2.png)
    - manual backprop ams grad

    feed-forward step:
        from input x we calculate the mean of every dimension in the feature space and then subtract this vector of mean values from every training example.
        we also calculate the per-dimension variance 
        we multiply by gamma and add beta to get the output!
    backward step:
        ![](batchnorm3.png)

- tensorflow broadcasting:
    ![](broadcast.png)

- tf.nn.moments(x, axes, shift=None, keepdims=False, name=None) is to calculate mean and variance

- to see coverage of unittests: conda install coverage
