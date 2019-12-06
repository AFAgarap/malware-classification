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

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def save_labels(predictions, actual, result_path, phase, model, step):
    """Saves the actual and predicted labels to a NPY file

    :param predictions: The NumPy array containing the predicted labels.
    :param actual: The NumPy array containing the actual labels.
    :param result_path: The path where to save the concatenated actual and predicted labels.
    :param phase: The phase for which the predictions is, i.e. training/validation/testing.
    :param model: The name of the model used for classification
    :param step: The time step for the NumPy arrays.
    :return:
    """

    if not os.path.exists(path=result_path):
        os.mkdir(result_path)

    # Concatenate the predicted and actual labels
    labels = np.concatenate((predictions, actual), axis=1)

    # save every labels array to NPY file
    np.save(
        file=os.path.join(result_path, "{}-{}-{}.npy".format(phase, model, step)),
        arr=labels,
    )


def load_data(dataset, standardize=True):
    """

    :param dataset:
    :param standardize:
    :return:
    """

    features = dataset["arr"][:, 0]
    features = np.array([feature for feature in features])
    features = np.reshape(
        features, (features.shape[0], features.shape[1] * features.shape[2])
    )

    if standardize:
        features = StandardScaler().fit_transform(features)

    labels = dataset["arr"][:, 1]
    labels = np.array([label for label in labels])

    return features, labels


def one_hot_encode(labels):
    """

    :param labels:
    :return:
    """
    one_hot = np.zeros((labels.shape[0], labels.max() + 1))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    labels = one_hot
    labels[labels == 0] = -1
    return labels


def plot_confusion_matrix(phase, path, class_names):
    """Plots the confusion matrix using matplotlib.

    Parameter
    ---------
    phase : str
      String value indicating for what phase is the confusion matrix, i.e. training/validation/testing
    path : str
      Directory where the predicted and actual label NPY files reside
    class_names : str
      List consisting of the class names for the labels

    Returns
    -------
    conf : array, shape = [num_classes, num_classes]
      Confusion matrix
    accuracy : float
      Predictive accuracy
    """

    # list all the results files
    files = list_files(path=path)

    labels = np.array([])

    for file in files:
        labels_batch = np.load(file)
        labels = np.append(labels, labels_batch)

        if (files.index(file) / files.__len__()) % 0.2 == 0:
            print(
                "Done appending {}% of {}".format(
                    (files.index(file) / files.__len__()) * 100, files.__len__()
                )
            )

    labels = np.reshape(labels, newshape=(labels.shape[0] // 50, 50))

    print("Done appending NPY files.")

    # get the predicted labels
    predictions = labels[:, :25]

    # get the actual labels
    actual = labels[:, 25:]

    # create a TensorFlow session
    with tf.Session() as sess:

        # decode the one-hot encoded labels to single integer
        predictions = sess.run(tf.argmax(predictions, 1))
        actual = sess.run(tf.argmax(actual, 1))

    # get the confusion matrix based on the actual and predicted labels
    conf = confusion_matrix(y_true=actual, y_pred=predictions)

    # get the classification report on the actual and predicted labels
    report = classification_report(
        y_true=actual, y_pred=predictions, target_names=class_names
    )

    # create a confusion matrix plot
    plt.imshow(conf, cmap=plt.cm.Greys, interpolation="nearest")

    # set the plot title
    plt.title("Confusion Matrix for {} Phase".format(phase))

    # legend of intensity for the plot
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    # show the plot
    plt.show()

    # get the accuracy of the phase
    accuracy = accuracy_score(y_pred=predictions, y_true=actual)

    # return the confusion matrix, the accuracy, and the classification report
    return conf, accuracy, report


def list_files(path):
    """Returns a list of files

    Parameter
    ---------
    path : str
      A string consisting of a path containing files.

    Returns
    -------
    file_list : list
      A list of the files present in the given directory

    Examples
    --------
    >>> PATH = '/home/data'
    >>> list_files(PATH)
    >>> ['/home/data/file1', '/home/data/file2', '/home/data/file3']
    """

    file_list = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        file_list.extend(os.path.join(dir_path, filename) for filename in file_names)
    return file_list
