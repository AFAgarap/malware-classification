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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import tensorflow as tf


def svm_loss(labels, logits, num_classes, penalty_parameter, weight):
    """Returns the L2-SVM loss

    :param labels:
    :param logits:
    :param num_classes:
    :param penalty_parameter:
    :param weight:
    """
    regularization_loss = tf.reduce_mean(tf.square(weight))
    hinge_loss = tf.reduce_mean(
        tf.square(
            tf.maximum(
                tf.zeros([tf.shape(logits)[0], num_classes]), 1 - labels * logits
            )
        )
    )
    loss = regularization_loss + penalty_parameter * hinge_loss
    return loss
