Questions

- why input_image shape is (8,1024,3,1) i.e. each channel has 1 value? is it the intensity ?
- why the _variable_on_cpu call in variable_with_weight_decay? 

Interesting

- function _variable_with_weight_decay makes a variable and if this variable is for example a weight that we want to be included in the regularization (e.g. L2) term then it also creates a variable for the weight decay term (regarding this weight variable) and adds it to the loss collection. this is a collection to keep all our loss terms!