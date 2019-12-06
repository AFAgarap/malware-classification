# Copyright 2017 Abien Fred Agarap

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""2 Convolutional Layers with Max Pooling CNN"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import os
import sys
import tensorflow as tf
import time
from utils.data import save_labels
from utils.svm import svm_loss


class CNN:
    def __init__(
        self, alpha, batch_size, num_classes, penalty_parameter, sequence_length
    ):
        """Initializes the CNN-SVM model

        :param alpha: The learning rate to be used by the model.
        :param batch_size: The number of batches to use for training/validation/testing.
        :param num_classes: The number of classes in the dataset.
        :param penalty_parameter: The SVM C penalty parameter.
        :param sequence_length: The number of features in the dataset.
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.name = "CNN-SVM"
        self.num_classes = num_classes
        self.penalty_parameter = penalty_parameter
        self.sequence_length = sequence_length

        def __graph__():
            # [BATCH_SIZE, SEQUENCE_LENGTH]
            x_input = tf.placeholder(
                dtype=tf.float32, shape=[None, sequence_length], name="x_input"
            )

            # [BATCH_SIZE, NUM_CLASSES]
            y_input = tf.placeholder(
                dtype=tf.float32, shape=[None, num_classes], name="y_input"
            )

            # First convolutional layer
            first_conv_weight = self.weight_variable([5, 5, 1, 36])
            first_conv_bias = self.bias_variable([36])

            input_image = tf.reshape(x_input, [-1, 32, 32, 1])

            first_conv_activation = tf.nn.leaky_relu(
                self.conv2d(input_image, first_conv_weight) + first_conv_bias
            )
            first_conv_pool = self.max_pool_2x2(first_conv_activation)

            # Second convolutional layer
            second_conv_weight = self.weight_variable([5, 5, 36, 72])
            second_conv_bias = self.bias_variable([72])

            second_conv_activation = tf.nn.leaky_relu(
                self.conv2d(first_conv_pool, second_conv_weight) + second_conv_bias
            )
            second_conv_pool = self.max_pool_2x2(second_conv_activation)

            # Fully-connected layer (Dense Layer)
            dense_layer_weight = self.weight_variable([8 * 8 * 72, 1024])
            dense_layer_bias = self.bias_variable([1024])

            second_conv_pool_flatten = tf.reshape(second_conv_pool, [-1, 8 * 8 * 72])
            dense_layer_activation = tf.nn.leaky_relu(
                tf.matmul(second_conv_pool_flatten, dense_layer_weight)
                + dense_layer_bias
            )

            # Dropout, to avoid overfitting
            keep_prob = tf.placeholder(tf.float32, name="p_keep")
            h_fc1_drop = tf.nn.dropout(dense_layer_activation, keep_prob)

            # Readout layer
            readout_weight = self.weight_variable([1024, num_classes])
            readout_bias = self.bias_variable([num_classes])

            logits = tf.matmul(h_fc1_drop, readout_weight) + readout_bias

            with tf.name_scope("loss"):
                loss = svm_loss(
                    labels=y_input,
                    logits=logits,
                    num_classes=num_classes,
                    penalty_parameter=penalty_parameter,
                    weight=readout_weight,
                )
            tf.summary.scalar("loss", loss)

            train_op = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)

            with tf.name_scope("metrics"):
                predicted_class = tf.sign(logits)
                predicted_class = tf.identity(predicted_class, name="predicted_class")
                with tf.name_scope("correct_prediction"):
                    correct_prediction = tf.equal(
                        tf.argmax(predicted_class, 1), tf.argmax(y_input, 1)
                    )
                with tf.name_scope("accuracy"):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            tf.summary.scalar("accuracy", accuracy)

            merged = tf.summary.merge_all()

            self.x_input = x_input
            self.y_input = y_input
            self.keep_prob = keep_prob
            self.logits = logits
            self.loss = loss
            self.train_op = train_op
            self.predicted_class = predicted_class
            self.accuracy = accuracy
            self.merged = merged

        sys.stdout.write("\n<log> Building graph...")
        __graph__()
        sys.stdout.write("</log>\n")

    def train(
        self,
        checkpoint_path,
        epochs,
        log_path,
        result_path,
        train_data,
        train_size,
        test_data,
        test_size,
    ):
        """Trains the initialized CNN-SVM model

        :param checkpoint_path: The path where to save the trained model.
        :param epochs: The number of passes through the entire dataset.
        :param log_path: The path where to save the TensorBoard summary reports.
        :param result_path:
        :param train_data: The data to be used for training the model.
        :param train_size: The size of training data.
        :param test_data: The data to be used for testing the model.
        :param test_size: The size of testing data.
        :return:
        """

        if not os.path.exists(path=checkpoint_path):
            os.mkdir(path=checkpoint_path)

        if not os.path.exists(path=log_path):
            os.mkdir(path=log_path)

        timestamp = str(time.asctime())

        saver = tf.train.Saver(max_to_keep=2)

        init = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )

        train_writer = tf.summary.FileWriter(
            logdir=os.path.join(log_path, timestamp + "-training"),
            graph=tf.get_default_graph(),
        )
        test_writer = tf.summary.FileWriter(
            logdir=os.path.join(log_path, timestamp + "-testing"),
            graph=tf.get_default_graph(),
        )

        with tf.Session() as sess:

            sess.run(init)

            checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

            if checkpoint and checkpoint.model_checkpoint_path:
                saver = tf.train.import_meta_graph(
                    checkpoint.model_checkpoint_path + ".meta"
                )
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
                print("Loaded {}".format(tf.train.latest_checkpoint(checkpoint_path)))

            try:
                for step in range(epochs * train_size // self.batch_size):

                    offset = (step * self.batch_size) % train_size

                    # train by batch
                    train_features = train_data[0][offset : (offset + self.batch_size)]
                    train_labels = train_data[1][offset : (offset + self.batch_size)]

                    # input dictionary with dropout of 50%
                    feed_dict = {
                        self.x_input: train_features,
                        self.y_input: train_labels,
                        self.keep_prob: 0.85,
                    }

                    # run the train op
                    train_summary, _, loss = sess.run(
                        [self.merged, self.train_op, self.loss], feed_dict=feed_dict
                    )

                    # every 100th step and at 0,
                    if step % 100 == 0:
                        # input dictionary without dropout
                        feed_dict = {
                            self.x_input: train_features,
                            self.y_input: train_labels,
                            self.keep_prob: 1.0,
                        }

                        # get the accuracy of training
                        train_accuracy = sess.run(self.accuracy, feed_dict=feed_dict)

                        # display the training accuracy
                        print(
                            "step: {}, training accuracy : {}, training loss : {}".format(
                                step, train_accuracy, loss
                            )
                        )

                        train_writer.add_summary(
                            summary=train_summary, global_step=step
                        )

                        saver.save(
                            sess=sess,
                            save_path=os.path.join(checkpoint_path, self.name),
                            global_step=step,
                        )
            except KeyboardInterrupt:
                print("Training interrupted at step {}".format(step))
                os._exit(1)
            finally:
                print("EOF -- Training done at step {}".format(step))

                for step in range(epochs * test_size // self.batch_size):

                    offset = (step * self.batch_size) % test_size

                    # train by batch
                    test_features = test_data[0][offset : (offset + self.batch_size)]
                    test_labels = test_data[1][offset : (offset + self.batch_size)]

                    # input dictionary
                    feed_dict = {
                        self.x_input: test_features,
                        self.y_input: test_labels,
                        self.keep_prob: 1.0,
                    }

                    # run the train op
                    test_summary, accuracy, loss, prediction, actual = sess.run(
                        [
                            self.merged,
                            self.accuracy,
                            self.loss,
                            self.predicted_class,
                            self.y_input,
                        ],
                        feed_dict=feed_dict,
                    )

                    # every 100th step and at 0,
                    if step % 100 == 0:

                        # display the testing accuracy and loss
                        print(
                            "step: {}, testing accuracy : {}, testing loss : {}".format(
                                step, accuracy, loss
                            )
                        )

                        test_writer.add_summary(summary=test_summary, global_step=step)

                    save_labels(
                        predictions=prediction,
                        actual=actual,
                        result_path=result_path,
                        phase="testing",
                        model=self.name,
                        step=step,
                    )

                print("EOF -- Testing done at step {}".format(step))

    @staticmethod
    def weight_variable(shape):
        """Returns a weight matrix consisting of arbitrary values.

        :param shape: The shape of the weight matrix to create.
        :return: The weight matrix consisting of arbitrary values.
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """Returns a bias matrix consisting of 0.1 values.

        :param shape: The shape of the bias matrix to create.
        :return: The bias matrix consisting of 0.1 values.
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        """Produces a convolutional layer that filters an image subregion

        :param x: The layer input.
        :param W: The size of the layer filter.
        :return: Returns a convolutional layer.
        """
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    @staticmethod
    def max_pool_2x2(x):
        """Downnsamples the image based on convolutional layer

        :param x: The input to downsample.
        :return: Downsampled input.
        """
        return tf.nn.max_pool(
            x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
