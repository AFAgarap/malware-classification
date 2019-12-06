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

"""Module to display the experiment results"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import argparse
from utils import data

MALWARE_FAMILIES = [
    "Adialer.C",
    "Agent.FYI",
    "Allaple.A",
    "Allaple.L",
    "Alueron.gen!J",
    "Autorun.K",
    "C2LOP.P",
    "C2LOP.gen!g",
    "Dialplatform.B",
    "Dontovo.A",
    "Fakerean",
    "Instantaccess",
    "Lolyda.AA1",
    "Lolyda.AA2",
    "Lolyda.AA3",
    "Lolyda.AT",
    "Malex.gen!J",
    "Obfuscator.AD",
    "Rbot!gen",
    "Skintrim.N",
    "Swizzor.gen!E",
    "Swizzor.gen!I",
    "VB.AT",
    "Wintrim.BX",
    "Yuner.A",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep Learning Using Support Vector Machine for Malware Classification"
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-r",
        "--result_path",
        required=True,
        type=str,
        help="path where the actual and predicted labels array files are saved",
    )
    group.add_argument(
        "-t",
        "--figure_title",
        required=True,
        type=str,
        help="the title of the confusion matrix figure",
    )
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    conf, acc, report = data.plot_confusion_matrix(
        arguments.figure_title, arguments.result_path, MALWARE_FAMILIES
    )

    print("{} Classification report :\n{}".format(arguments.figure_title, report))

    print("{} Confusion matrix :\n{}".format(arguments.figure_title, conf))

    print("{} Accuracy : {}".format(arguments.figure_title, acc))


if __name__ == "__main__":
    args = parse_args()

    main(arguments=args)
