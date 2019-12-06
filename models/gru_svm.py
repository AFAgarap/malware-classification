# A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and
# Support Vector Machine (SVM) for Intrusion Detection in Network Traffic Data
# Copyright (C) 2017  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""Implementation of the GRU-SVM model [http://arxiv.org/abs/1709.03082] by A.F. Agarap"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.3.9"
__author__ = "Abien Fred Agarap"

import numpy as np
import os
import sys
import tensorflow as tf
import time
from utils.data import save_labels
from utils.svm import svm_loss


class GruSvm:
    """Implementation of the GRU-SVM model using TensorFlow"""

    def __init__(
        self,
        alpha,
        batch_size,
        cell_size,
        dropout_rate,
        num_classes,
        num_layers,
        sequence_height,
        sequence_width,
        svm_c,
    ):
        """Initialize the GRU-SVM model.

        :param alpha: The learning rate for the GRU-SVM model.
        :param batch_size: The number of data per batch to use for training/testing.
        :param cell_size: The size of cell state.
        :param dropout_rate: The dropout rate to be used.
        :param num_classes: The number of classes in a dataset.
        :param num_layers: The number of hidden layers
        :param sequence_height: The height dimension of the image.
        :param sequence_width: The width dimension of the width.
        :param svm_c: The SVM penalty parameter C.
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.cell_size = cell_size
        self.dropout_rate = dropout_rate
        self.name = "GRU-SVM"
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.sequence_height = sequence_height
        self.sequence_width = sequence_width
        self.svm_c = svm_c

        def __graph__():
            """Build the inference graph"""
            with tf.name_scope("input"):
                # [BATCH_SIZE, SEQUENCE_WIDTH, SEQUENCE_HEIGHT]
                x_input = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, self.sequence_width, sequence_height],
                    name="x_input",
                )

                # [BATCH_SIZE, NUM_CLASSES]
                y_input = tf.placeholder(
                    dtype=tf.float32, shape=[None, self.num_classes], name="y_input"
                )

            state = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.cell_size * self.num_layers],
                name="initial_state",
            )

            p_keep = tf.placeholder(dtype=tf.float32, name="p_keep")
            learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

            cells = [
                tf.contrib.rnn.GRUCell(self.cell_size) for _ in range(self.num_layers)
            ]
            drop_cells = [
                tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=p_keep)
                for cell in cells
            ]
            multi_cell = tf.contrib.rnn.MultiRNNCell(drop_cells, state_is_tuple=False)
            multi_cell = tf.contrib.rnn.DropoutWrapper(
                multi_cell, output_keep_prob=p_keep
            )

            # outputs: [BATCH_SIZE, SEQUENCE_LENGTH, CELL_SIZE]
            # states: [BATCH_SIZE, CELL_SIZE]
            outputs, states = tf.nn.dynamic_rnn(
                multi_cell, x_input, initial_state=state, dtype=tf.float32
            )

            states = tf.identity(states, name="H")

            with tf.name_scope("final_training_ops"):
                with tf.name_scope("weights"):
                    xav_init = tf.contrib.layers.xavier_initializer
                    weight = tf.get_variable(
                        name="weights",
                        shape=[self.cell_size, self.num_classes],
                        initializer=xav_init(),
                    )
                    self.variable_summaries(weight)
                with tf.name_scope("biases"):
                    bias = tf.get_variable(
                        name="biases",
                        initializer=tf.constant(0.1, shape=[self.num_classes]),
                    )
                    self.variable_summaries(bias)
                hf = tf.transpose(outputs, [1, 0, 2])
                last = tf.gather(hf, int(hf.get_shape()[0]) - 1)
                with tf.name_scope("Wx_plus_b"):
                    output = tf.matmul(last, weight) + bias
                    tf.summary.histogram("pre-activations", output)

            # L2-SVM
            with tf.name_scope("loss"):
                loss = svm_loss(
                    labels=y_input,
                    logits=output,
                    num_classes=num_classes,
                    penalty_parameter=self.svm_c,
                    weight=weight,
                )
            tf.summary.scalar("loss", loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss
            )

            with tf.name_scope("accuracy"):
                predicted_class = tf.sign(output)
                predicted_class = tf.identity(predicted_class, name="predicted_class")
                with tf.name_scope("correct_prediction"):
                    correct = tf.equal(
                        tf.argmax(predicted_class, 1), tf.argmax(y_input, 1)
                    )
                with tf.name_scope("accuracy"):
                    accuracy = tf.reduce_mean(tf.cast(correct, "float"))
            tf.summary.scalar("accuracy", accuracy)

            # merge all the summaries collected from the TF graph
            merged = tf.summary.merge_all()

            # set class properties
            self.x_input = x_input
            self.y_input = y_input
            self.p_keep = p_keep
            self.loss = loss
            self.optimizer = optimizer
            self.state = state
            self.states = states
            self.learning_rate = learning_rate
            self.predicted_class = predicted_class
            self.accuracy = accuracy
            self.merged = merged

        sys.stdout.write("\n<log> Building Graph...")
        __graph__()
        sys.stdout.write("</log>\n")

    def train(
        self,
        checkpoint_path,
        log_path,
        epochs,
        train_data,
        train_size,
        test_data,
        test_size,
        result_path,
    ):
        """

        :param checkpoint_path: The path where to save the trained model.
        :param log_path: The path where to save the TensorBoard summaries.
        :param epochs: The number of passes through the whole dataset.
        :param train_data: The NumPy array training dataset.
        :param train_size: The size of `train_data`.
        :param test_data: The NumPy array testing dataset.
        :param test_size: The size of `test_data`.
        :param result_path: The path where to save the actual and predicted classes array.
        :return:
        """

        if not os.path.exists(path=checkpoint_path):
            os.mkdir(path=checkpoint_path)

        if not os.path.exists(path=log_path):
            os.mkdir(path=log_path)

        saver = tf.train.Saver(max_to_keep=2)

        # initialize H (current_state) with values of zeros
        current_state = np.zeros([self.batch_size, self.cell_size * self.num_layers])

        # variables initializer
        init_op = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )

        # get the time tuple
        timestamp = str(time.asctime())

        train_writer = tf.summary.FileWriter(
            logdir=log_path + timestamp + "-training", graph=tf.get_default_graph()
        )
        test_writer = tf.summary.FileWriter(
            logdir=log_path + timestamp + "-testing", graph=tf.get_default_graph()
        )

        with tf.Session() as sess:

            sess.run(init_op)

            checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

            if checkpoint and checkpoint.model_checkpoint_path:
                saver = tf.train.import_meta_graph(
                    checkpoint.model_checkpoint_path + ".meta"
                )
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
                print(
                    "Loading {}".format(
                        tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path)
                    )
                )

            try:
                for step in range(epochs * train_size // self.batch_size):

                    # set the value for slicing
                    # e.g. step = 0, batch_size = 256, train_size = 1898240
                    # (0 * 256) % 1898240 = 0
                    # [offset:(offset + batch_size)] = [0:256]
                    offset = (step * self.batch_size) % train_size
                    train_example_batch = train_data[0][
                        offset : (offset + self.batch_size)
                    ]
                    train_label_batch = train_data[1][
                        offset : (offset + self.batch_size)
                    ]

                    # dictionary for key-value pair input for training
                    feed_dict = {
                        self.x_input: train_example_batch,
                        self.y_input: train_label_batch,
                        self.state: current_state,
                        self.learning_rate: self.alpha,
                        self.p_keep: self.dropout_rate,
                    }

                    train_summary, _, predictions, actual, next_state = sess.run(
                        [
                            self.merged,
                            self.optimizer,
                            self.predicted_class,
                            self.y_input,
                            self.states,
                        ],
                        feed_dict=feed_dict,
                    )

                    # Display training loss and accuracy every 100 steps and at step 0
                    if step % 100 == 0:
                        # get train loss and accuracy
                        train_loss, train_accuracy = sess.run(
                            [self.loss, self.accuracy], feed_dict=feed_dict
                        )

                        # display train loss and accuracy
                        print(
                            "step [{}] train -- loss : {}, accuracy : {}".format(
                                step, train_loss, train_accuracy
                            )
                        )

                        # write the train summary
                        train_writer.add_summary(train_summary, step)

                        # save the model at current step
                        saver.save(
                            sess,
                            os.path.join(checkpoint_path, self.name),
                            global_step=step,
                        )

                    current_state = next_state

                    save_labels(
                        predictions=predictions,
                        actual=actual,
                        result_path=result_path,
                        step=step,
                        model=self.name,
                        phase="training",
                    )

            except KeyboardInterrupt:
                print("Training interrupted at {}".format(step))
                os._exit(1)
            finally:
                print("EOF -- Training done at step {}".format(step))

                for step in range(epochs * test_size // self.batch_size):

                    offset = (step * self.batch_size) % test_size
                    test_example_batch = test_data[0][
                        offset : (offset + self.batch_size)
                    ]
                    test_label_batch = test_data[1][offset : (offset + self.batch_size)]

                    # dictionary for key-value pair input for testing
                    feed_dict = {
                        self.x_input: test_example_batch,
                        self.y_input: test_label_batch,
                        self.state: np.zeros(
                            [self.batch_size, self.cell_size * self.num_layers]
                        ),
                        self.p_keep: 1.0,
                    }

                    (
                        test_summary,
                        predictions,
                        actual,
                        test_loss,
                        test_accuracy,
                    ) = sess.run(
                        [
                            self.merged,
                            self.predicted_class,
                            self.y_input,
                            self.loss,
                            self.accuracy,
                        ],
                        feed_dict=feed_dict,
                    )

                    # Display test loss and accuracy every 100 steps
                    if step % 100 == 0 and step > 0:

                        # add the test summary
                        test_writer.add_summary(summary=test_summary, global_step=step)

                        # display test loss and accuracy
                        print(
                            "step [{}] test -- loss : {}, accuracy : {}".format(
                                step, test_loss, test_accuracy
                            )
                        )

                    save_labels(
                        predictions=predictions,
                        actual=actual,
                        result_path=result_path,
                        step=step,
                        model=self.name,
                        phase="testing",
                    )

                print("EOF -- Testing done at step {}".format(step))

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
