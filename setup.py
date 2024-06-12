# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="emg2qwerty",
    version="0.1.0",
    description="Baselines for modeling QWERTY typing from surface electromyography.",
    author="Viswanath Sivakumar",
    author_email="viswanath@meta.com",
    packages=find_packages(),
    install_requires=[
        # Use environment.yml to create conda env
    ],
)
