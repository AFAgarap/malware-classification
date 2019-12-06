# Copyright 2017 Abien Fred Agarap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""Implementation of the MLP-SVM model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import os
import sys
import time
import tensorflow as tf
from utils.data import save_labels
from utils.svm import svm_loss


class MLP:
    """Implementation of the MLP-SVM model using TensorFlow"""

    def __init__(
        self, alpha, batch_size, node_size, num_classes, num_features, penalty_parameter
    ):
        """Initialize the MLP-SVM model

        :param alpha: The learning rate to be used by the neural network.
        :param batch_size: The number of batches to use for training/validation/testing.
        :param node_size: The number of neurons in the neural network.
        :param num_classes: The number of classes in a dataset.
        :param num_features: The number of features in a dataset.
        :param penalty_parameter: The SVM C parameter.
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.name = "MLP-SVM"
        self.node_size = node_size
        self.num_classes = num_classes
        self.num_features = num_features
        self.penalty_parameter = penalty_parameter

        def __graph__():
            """Build the inference graph"""

            with tf.name_scope("input"):
                # [BATCH_SIZE, NUM_FEATURES]
                x_input = tf.placeholder(
                    dtype=tf.float32, shape=[None, self.num_features], name="x_input"
                )

                # [BATCH_SIZE, NUM_CLASSES]
                y_input = tf.placeholder(
                    dtype=tf.float32, shape=[None, self.num_classes], name="y_input"
                )

            learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

            first_hidden_layer = {
                "weights": self.weight_variable(
                    "h1_w_layer", [self.num_features, self.node_size[0]]
                ),
                "biases": self.bias_variable("h1_b_layer", [self.node_size[0]]),
            }

            second_hidden_layer = {
                "weights": self.weight_variable(
                    "h2_w_layer", [self.node_size[0], self.node_size[1]]
                ),
                "biases": self.bias_variable("h2_b_layer", [self.node_size[1]]),
            }

            third_hidden_layer = {
                "weights": self.weight_variable(
                    "h3_w_layer", [self.node_size[1], self.node_size[2]]
                ),
                "biases": self.bias_variable("h3_b_layer", [self.node_size[2]]),
            }

            last_hidden_layer = {
                "weights": self.weight_variable(
                    "output_w_layer", [self.node_size[2], self.num_classes]
                ),
                "biases": self.bias_variable("output_b_layer", [self.num_classes]),
            }

            first_layer = (
                tf.matmul(x_input, first_hidden_layer["weights"])
                + first_hidden_layer["biases"]
            )
            first_layer = tf.nn.leaky_relu(first_layer)

            second_layer = (
                tf.matmul(first_layer, second_hidden_layer["weights"])
                + second_hidden_layer["biases"]
            )
            second_layer = tf.nn.leaky_relu(second_layer)

            third_layer = (
                tf.matmul(second_layer, third_hidden_layer["weights"])
                + third_hidden_layer["biases"]
            )
            third_layer = tf.nn.leaky_relu(third_layer)

            output_layer = (
                tf.matmul(third_layer, last_hidden_layer["weights"])
                + last_hidden_layer["biases"]
            )
            tf.summary.histogram("pre-activations", output_layer)

            with tf.name_scope("loss"):
                loss = svm_loss(
                    labels=y_input,
                    logits=output_layer,
                    num_classes=self.num_classes,
                    penalty_parameter=self.penalty_parameter,
                    weight=last_hidden_layer["weights"],
                )
            tf.summary.scalar("loss", loss)

            optimizer_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss
            )

            with tf.name_scope("accuracy"):
                predicted_class = tf.sign(output_layer)
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
            self.learning_rate = learning_rate
            self.loss = loss
            self.optimizer_op = optimizer_op
            self.predicted_class = predicted_class
            self.accuracy = accuracy
            self.merged = merged

        sys.stdout.write("\n<log> Building Graph...")
        __graph__()
        sys.stdout.write("</log>\n")

    def train(
        self,
        checkpoint_path,
        num_epochs,
        log_path,
        train_data,
        train_size,
        test_data,
        test_size,
        result_path,
    ):
        """Trains the initialized MLP-SVM model.

        :param checkpoint_path: The path where to save the trained model.
        :param num_epochs: The number of passes over the entire dataset.
        :param log_path: The path where to save the TensorBoard logs.
        :param train_data: The NumPy array to be used as training dataset.
        :param train_size: The size of the `train_data`.
        :param test_data: The NumPy array to be used as testing dataset.
        :param test_size: The size of the `test_data`.
        :param result_path: The path where to save the actual and predicted labels, for later analysis
        :return:
        """

        if not os.path.exists(path=checkpoint_path):
            os.mkdir(path=checkpoint_path)

        if not os.path.exists(path=log_path):
            os.mkdir(path=log_path)

        timestamp = str(time.asctime())

        saver = tf.train.Saver(max_to_keep=2)
        # initialize the variables
        init_op = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )

        train_writer = tf.summary.FileWriter(
            log_path + timestamp + "-training", graph=tf.get_default_graph()
        )
        test_writer = tf.summary.FileWriter(
            log_path + timestamp + "-test", graph=tf.get_default_graph()
        )

        with tf.Session() as sess:

            sess.run(init_op)

            checkpoint = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_path)

            if checkpoint and checkpoint.model_checkpoint_path:
                saver = tf.train.import_meta_graph(
                    checkpoint.model_checkpoint_path + ".meta"
                )
                saver.restore(
                    sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path)
                )
                print(
                    "Loaded {}".format(
                        tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path)
                    )
                )

            try:
                for step in range(num_epochs * train_size // self.batch_size):

                    offset = (step * self.batch_size) % train_size

                    # train by batch
                    train_data_batch = train_data[0][
                        offset : (offset + self.batch_size)
                    ]
                    train_label_batch = train_data[1][
                        offset : (offset + self.batch_size)
                    ]

                    feed_dict = {
                        self.x_input: train_data_batch,
                        self.y_input: train_label_batch,
                        self.learning_rate: self.alpha,
                    }

                    train_summary, _, step_loss = sess.run(
                        [self.merged, self.optimizer_op, self.loss], feed_dict=feed_dict
                    )

                    if step % 100 == 0 and step > 0:

                        # get the training accuracy at current time step
                        train_accuracy = sess.run(self.accuracy, feed_dict=feed_dict)

                        # display the training accuracy
                        print(
                            "step [{}] train -- loss : {}, accuracy : {}".format(
                                step, step_loss, train_accuracy
                            )
                        )

                        # add TensorBoard summary
                        train_writer.add_summary(train_summary, global_step=step)

                        # save the model at this time step
                        saver.save(
                            sess=sess,
                            save_path=os.path.join(checkpoint_path, self.name),
                            global_step=step,
                        )

            except KeyboardInterrupt:
                print("KeyboardInterrupt at step {}".format(step))
                os._exit(1)
            finally:
                print("EOF -- Training done at step {}".format(step))

                for step in range(num_epochs * test_size // self.batch_size):

                    offset = (step * self.batch_size) % test_size

                    # test by batch
                    test_data_batch = test_data[0][offset : (offset + self.batch_size)]
                    test_label_batch = test_data[1][offset : (offset + self.batch_size)]

                    feed_dict = {
                        self.x_input: test_data_batch,
                        self.y_input: test_label_batch,
                    }

                    (
                        test_summary,
                        test_accuracy,
                        test_loss,
                        predictions,
                        actual,
                    ) = sess.run(
                        [
                            self.merged,
                            self.accuracy,
                            self.loss,
                            self.predicted_class,
                            self.y_input,
                        ],
                        feed_dict=feed_dict,
                    )

                    if step % 100 == 0 and step > 0:

                        # display testing loss and accuracy
                        print(
                            "step [{}] test -- loss : {}, accuracy : {}".format(
                                step, test_loss, test_accuracy
                            )
                        )

                        # add TensorBoard summary
                        test_writer.add_summary(test_summary, step)

                    save_labels(
                        predictions=predictions,
                        actual=actual,
                        result_path=result_path,
                        phase="testing",
                        model=self.name,
                        step=step,
                    )

                print("EOF -- Testing done at step {}".format(step))

    @staticmethod
    def weight_variable(name, shape):
        """Returns a weight matrix consisting of arbitrary values.

        :param name: The name for the `weight` tensor.
        :param shape: The shape of the weight matrix to create.
        :return: The weight matrix consisting of arbitrary values.
        """
        xav_init = tf.contrib.layers.xavier_initializer
        return tf.get_variable(name=name, shape=shape, initializer=xav_init())

    @staticmethod
    def bias_variable(name, shape):
        """Returns a bias matrix consisting of 0.1 values.

        :param name: The name for the `bias` tensor.
        :param shape: The shape of the bias matrix to create.
        :return: The bias matrix consisting of 0.1 values.
        """
        initial_value = tf.constant([0.1], shape=shape)
        return tf.get_variable(name=name, initializer=initial_value)

    @staticmethod
    def variable_summaries(var):
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean", mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev)
            tf.summary.scalar("max", tf.reduce_max(var))
            tf.summary.scalar("min", tf.reduce_min(var))
            tf.summary.histogram("histogram", var)
