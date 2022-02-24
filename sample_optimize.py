# -*- coding: utf-8 -*-
import logging
import os
import sys

import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

from nn_regressor import dump_yaml, load_yaml, optimize


def compute_metric(result: pd.DataFrame) -> float:
    y_true = result["y_true"].values
    y_pred = result["y_pred"].values
    r, p = scipy.stats.pearsonr(y_pred, y_true)
    return r


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S%p",
    )

    logger = logging.getLogger(__name__)
    yaml_path = os.path.join(os.getcwd(), "models.yaml")
    confs_nn = load_yaml(yaml_path)

    dataset = pd.read_csv("sample_dataset.csv")
    model_list = set(dataset["model_id"].values)

    for name in model_list:

        x = dataset.loc[dataset["model_id"] == name].drop(columns="target")
        y = dataset["target"].loc[dataset["model_id"] == name]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.1, shuffle=False
        )

        optimal = optimize(
            generations=5,  # Number of times to evolve the population.
            population=20,  # Number of networks in each generation.
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            actual_best=confs_nn.get(str(name), confs_nn["default"]),
            confs_nn=confs_nn,
            compute_metric=compute_metric,
        )

        optimal_par = optimal[0].parameters
        confs_nn = load_yaml(yaml_path)
        confs_nn[str(name)] = optimal_par
        dump_yaml(yaml_path)

    logger.info("...MODEL FINISHED!!!")
