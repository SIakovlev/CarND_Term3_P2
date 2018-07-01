import os.path
import tensorflow as tf
import helper
import warnings
import matplotlib.pyplot as plt
import numpy as np
from distutils.version import LooseVersion
from datetime import datetime
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, 
    layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # add scaling in order to train the network all at once
    #vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')
    #vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')

    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')
    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')
    
    # Add 1x1 convolution to VGG16 as the last layer instead of fully connected layer
    fcn8_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, (1, 1), 1, 
        padding="same", kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    # Upsample it with the output shape corresponding to the 4th layer output
    fcn9 = tf.layers.conv2d_transpose(fcn8_1x1, vgg_layer4_out_scaled.get_shape().as_list()[-1], 
                                      (4, 4), 2, padding="same", 
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    # Skip connections implementation
    sum_9_4 = tf.add(fcn9, vgg_layer4_out_scaled)
    # Upsample it twice again with the output shape corresponding to the 3rd layer output
    sum_9_4 = tf.layers.conv2d_transpose(sum_9_4, vgg_layer3_out_scaled.get_shape().as_list()[-1], 
                                         (4, 4), 2, padding="same", 
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    # Skip connections implementation
    out = tf.add(sum_9_4, vgg_layer3_out_scaled)
    # 8x upsampling to the final output layer
    out = tf.layers.conv2d_transpose(out, num_classes, (16, 16), 8, padding="same", 
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    return out
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
                                                                  labels=correct_label_reshaped))
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.001  # Choose an appropriate one.
    loss += reg_constant * sum(reg_losses)
    
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    return logits, train_op, loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, iou, iou_op, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data. Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param iou: IoU metric, calculated using iou_op 
    :param iou_op: TF Operation to calculate IoU metric
    :param loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    keep_prob_value = 0.75
    learning_rate_value = 0.0001
    for epoch in range(epochs):
        total_loss = 0
        total_iou = 0
        counter = 0
        for image_batch, label_batch in get_batches_fn(batch_size):
            loss_batch, _, _ = sess.run([loss, train_op, iou_op], 
                feed_dict={input_image: image_batch, correct_label: label_batch,
                           keep_prob: keep_prob_value, learning_rate: learning_rate_value})
            total_loss += loss_batch
            total_iou += sess.run(iou)
            counter += 1

        print("Epoch: {}".format(epoch + 1))
        print("Accuracy IoU: {}%".format(total_iou/counter*100))  
        print("Loss: {:.3f}".format(total_loss))
        print()

tests.test_train_nn(train_nn)

def mean_iou(logits, correct_label, num_classes):
    temp = tf.cast(tf.greater(tf.nn.softmax(logits), tf.constant(0.5, dtype=tf.float32)), 
                   dtype=tf.float32)
    nn_out_reshaped = tf.reshape(temp, (-1, num_classes))
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))
    # Compute the mean IoU
    iou, iou_op = tf.metrics.mean_iou(correct_label_reshaped, nn_out_reshaped, num_classes)
    return iou, iou_op

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    models_dir = './models'
    tests.test_for_kitti_dataset(data_dir)

    EPOCHS = 100
    BATCH_SIZE = 16
    learning_rate = tf.placeholder(tf.float32)
    correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        fcn_output = layers(layer3, layer4, layer7, num_classes)
        logits, train_op, loss = optimize(fcn_output, correct_label, learning_rate, num_classes)
        iou, iou_op = mean_iou(logits, correct_label, num_classes)
        # TODO: Train NN using the train_nn function
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        print()
        print("Model build successful, starting training...")

        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, loss, iou, iou_op, 
                 image_input, correct_label, keep_prob, learning_rate)
        
        print("Saving current model...")
        now = datetime.utcnow().strftime("%S%M%H%d%m%Y")
        model_dir = "{}/run-{}/model".format(models_dir, now)
        saver.save(sess, model_dir)
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)
        # print(sess.run(logits, feed_dict={input_image: test_im[0]}))
        
        print("Done!")
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
