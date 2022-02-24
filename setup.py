# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="nn_regressor",
    version="0.1.0",
    description="Wrapper around Keras NN for simple regression on tabular data",
    url="https://github.com/AlessandroGianfelici/NN_regressor",
    author="Alessandro Gianfelici",
    author_email="alessandro.gianfelici@hotmail.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "pyyaml",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "tensorflow",
        "pre-commit",
        "isort",
        "black",
        "pampy",
    ],
    zip_safe=False,
)
