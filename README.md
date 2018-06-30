[image1]: ./pics/1.png
[image2]: ./pics/2.png
[image3]: ./pics/3.png
[image4]: ./pics/4.png
[image5]: ./pics/5.png

# CarND_Term3_P2
Image Segmentation Project

### Project criterias
In this section I address all the necessary criterias

####  Build the Neural Network
Funcitons `load_vgg`, `layers`, `optimize` and `train_nn` pass tests successfully.

#### Neural Network Training
* Model decreases loss over time:
  ```
  Model build successful, starting training...
  Epoch: 1
  Accuracy IoU: 56.84587422170138%
  Loss: 6.976
  
  Epoch: 2
  Accuracy IoU: 75.3019950891796%
  Loss: 2.786
  
  ...
  
  Epoch: 99
  Accuracy IoU: 98.01804109623558%
  Loss: 0.123
  
  Epoch: 100
  Accuracy IoU: 98.03287763344613%
  Loss: 0.133
  ```
* Parameters choice and NN tuning:
  * `batch_size` was chosen to be `16` based on my GPU configuration (GTX1070, 8GB). Any number larger than this one lead to a lack GPU memory.
  * `epochs = 100`. After around 100 epochs loss did not reduce significantly
  * Dropout was used in order to prevent overfitting. `keep_prob` value was chosen to be `0.75`.
  * Another technique that was used for overfitting prevention is L2 regularisation. Importantly, it should be included in both layer declarations and loss function:
    * in layers:
      ```python
      ...
      out = tf.layers.conv2d_transpose(out, num_classes, (16, 16), 8, padding="same", 
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
      ...
      ```
    * in loss function: 
      ```python
      reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      reg_constant = 0.001  # Choose an appropriate one.
      loss += reg_constant * sum(reg_losses)
      ```
  * In order to improve model performance, 3rd and 4th layers of the VGG network were scaled by factors of `0.0001` and `0.01`:
    ```python
    # add scaling in order to train the network all at once
    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')
    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')
    ```
* Does the project correctly labels the road?
  * 
  <p float="left">
    <img src="/pics/1.png" width="270" />
    <img src="/pics/2.png" width="270" /> 
    <img src="/pics/3.png" width="270" />
  </p>
  <p float="left">
    <img src="/pics/4.png" width="270" />
    <img src="/pics/5.png" width="270" /> 
    <img src="/pics/6.png" width="270" />
  </p>
  * A solution that is close to best would label at least 80% of the road and label no more than 20% of non-road pixels as road. _specify test set IoU accuracy_


