Questions

- why input_image shape is (8,1024,3,1) i.e. each channel has 1 value? is it the intensity ?
- why the _variable_on_cpu call in variable_with_weight_decay? maybe they specify all variables on cpu so that in case you run on multi gpu it is more efficient to have everything on cpu..(https://stackoverflow.com/questions/34428850/variables-on-cpu-training-gradients-on-gpu)
  check also: https://jhui.github.io/2017/03/07/TensorFlow-GPU/
  depends on hardware configuration...e.g. Tesla K80: If the GPUs are on the same PCI Express and are able to communicate using NVIDIA GPUDirect Peer to Peer, we place the variables equally across the GPUs. Otherwise, we place the variables on the CPU. Titan X, P100: For models like ResNet and InceptionV3, placing variables on the CPU. But for models with a lot of variables like AlexNet and VGG, using GPUs with NCCL is better.

Interesting

- function _variable_with_weight_decay makes a variable and if this variable is for example a weight that we want to be included in the regularization (e.g. L2) term then it also creates a variable for the weight decay term (regarding this weight variable) and adds it to the loss collection. this is a collection to keep all our loss terms!

- There are two types of parallelism:
    - Model parallelism - Different GPUs run different part of the code. Batches of data pass through all GPUs.
    e.g.
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
    
    - Data parallelism - We use multiple GPUs to run the same TensorFlow code. Each GPU is feed with different batch of data.
      <br>
      If a host have multiple GPUs with the same memory and computation capacity, it will be simpler to scale with data parallelism.
      e.g.
      ```python
     import tensorflow as tf
    
    c = []
    a = tf.get_variable(f"a", [2, 2, 3], initializer=tf.random_uniform_initializer(-1, 1))
    b = tf.get_variable(f"b", [2, 3, 2], initializer=tf.random_uniform_initializer(-1, 1))
    
    # Multiple towers
    for i, d in enumerate(['/gpu:0', '/gpu:1']):
    with tf.device(d):
        c.append(tf.matmul(a[i], b[i]))   # Tower i is responsible for batch data i.
    
    with tf.device('/cpu:0'):
    sum = tf.add_n(c)
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    print(sess.run(sum))
    ```
  
  - the conv2d has an arguments strides of shape 1,2,or 4.
  if 4 then it is stride for [N,H,W,C] i.e. stride for batch, height, width and channel. normally you set N to 1 (you don't want to skip any batch otherwise you wouldnt include those data in training) and C to 1 (you don't want to skip data - channel information).
  

Notes

- TensorFlow variables are used to share and persist some stats that are manipulated by our program. That is, when you define a variable, TensorFlow adds a tf.Operation to your graph. Then, this operation will store a writable tensor value that persists between tf.Session.run calls. So, you can update the value of a variable through each run, while you cannot update tensor (e.g a tensor created by tf.constant()) through multiple runs in a session.
To define variables we use the command tf.Variable(). To be able to use variables in a computation graph it is necessary to initialize them before running the graph in a session. This is done by running tf.global_variables_initializer().

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
