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

"""Main program implementing the deep learning algorithms"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import argparse
from models.cnn_svm import CNN
from models.gru_svm import GruSvm
from models.mlp_svm import MLP
import numpy as np
from sklearn.model_selection import train_test_split
from utils.data import load_data
from utils.data import one_hot_encode

BATCH_SIZE = 256
CELL_SIZE = 256
DROPOUT_RATE = 0.85
LEARNING_RATE = 1e-3
NODE_SIZE = [512, 256, 128]
NUM_LAYERS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep Learning Using Support Vector Machine for Malware Classification"
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-m",
        "--model",
        required=True,
        type=int,
        help="[1] CNN-SVM, [2] GRU-SVM, [3] MLP-SVM",
    )
    group.add_argument(
        "-d", "--dataset", required=True, type=str, help="the dataset to be used"
    )
    group.add_argument(
        "-n", "--num_epochs", required=True, type=int, help="number of epochs"
    )
    group.add_argument(
        "-c",
        "--penalty_parameter",
        required=True,
        type=float,
        help="the SVM C penalty parameter",
    )
    group.add_argument(
        "-k",
        "--checkpoint_path",
        required=True,
        type=str,
        help="path where to save the trained model",
    )
    group.add_argument(
        "-l",
        "--log_path",
        required=True,
        type=str,
        help="path where to save the TensorBoard logs",
    )
    group.add_argument(
        "-r",
        "--result_path",
        required=True,
        type=str,
        help="path where to save actual and predicted labels array",
    )
    arguments = parser.parse_args()
    return arguments


def main(arguments):

    model_choice = arguments.model
    assert (
        model_choice == 1 or model_choice == 2 or model_choice == 3
    ), "Invalid choice: Choose among 1, 2, and 3 only."

    dataset = np.load(arguments.dataset)

    features, labels = load_data(dataset=dataset)

    labels = one_hot_encode(labels=labels)

    # get the number of features
    num_features = features.shape[1]

    # get the number of classes
    num_classes = labels.shape[1]

    # split the dataset by 70/30
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.30, stratify=labels
    )

    train_size = int(train_features.shape[0])
    train_features = train_features[: train_size - (train_size % BATCH_SIZE)]
    train_labels = train_labels[: train_size - (train_size % BATCH_SIZE)]

    test_size = int(test_features.shape[0])
    test_features = test_features[: test_size - (test_size % BATCH_SIZE)]
    test_labels = test_labels[: test_size - (test_size % BATCH_SIZE)]

    if model_choice == 1:
        model = CNN(
            alpha=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            num_classes=num_classes,
            penalty_parameter=arguments.penalty_parameter,
            sequence_length=num_features,
        )
        model.train(
            checkpoint_path=arguments.checkpoint_path,
            log_path=arguments.log_path,
            result_path=arguments.result_path,
            epochs=arguments.num_epochs,
            train_data=[train_features, train_labels],
            train_size=int(train_features.shape[0]),
            test_data=[test_features, test_labels],
            test_size=int(test_features.shape[0]),
        )
    elif model_choice == 2:
        train_features = np.reshape(
            train_features,
            (
                train_features.shape[0],
                int(np.sqrt(train_features.shape[1])),
                int(np.sqrt(train_features.shape[1])),
            ),
        )
        test_features = np.reshape(
            test_features,
            (
                test_features.shape[0],
                int(np.sqrt(test_features.shape[1])),
                int(np.sqrt(test_features.shape[1])),
            ),
        )
        model = GruSvm(
            alpha=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            cell_size=CELL_SIZE,
            dropout_rate=DROPOUT_RATE,
            num_classes=num_classes,
            num_layers=NUM_LAYERS,
            sequence_height=train_features.shape[2],
            sequence_width=train_features.shape[1],
            svm_c=arguments.penalty_parameter,
        )
        model.train(
            checkpoint_path=arguments.checkpoint_path,
            log_path=arguments.log_path,
            epochs=arguments.num_epochs,
            train_data=[train_features, train_labels],
            train_size=int(train_features.shape[0]),
            test_data=[test_features, test_labels],
            test_size=int(test_features.shape[0]),
            result_path=arguments.result_path,
        )
    elif model_choice == 3:
        model = MLP(
            alpha=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            node_size=NODE_SIZE,
            num_classes=num_classes,
            num_features=num_features,
            penalty_parameter=arguments.penalty_parameter,
        )
        model.train(
            checkpoint_path=arguments.checkpoint_path,
            num_epochs=arguments.num_epochs,
            log_path=arguments.log_path,
            train_data=[train_features, train_labels],
            train_size=int(train_features.shape[0]),
            test_data=[test_features, test_labels],
            test_size=int(test_features.shape[0]),
            result_path=arguments.result_path,
        )


if __name__ == "__main__":
    args = parse_args()

    main(args)
