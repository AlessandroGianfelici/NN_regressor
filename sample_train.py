# -*- coding: utf-8 -*-
import logging
import os
import sys

import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

from nn_regressor import NeuralNetworkRegressor, load_yaml, select_or_create


def train(impianto, x_train, x_test, y_train, y_test, confs_nn, compute_metric):
    """
    Model training
    """
    nn_regressor = NeuralNetworkRegressor(
        confs_nn.get(str(impianto), confs_nn["default"])
    )
    nn_regressor.fit(x_train, y_train, batch_size=512, verbose=0)
    result_test = pd.DataFrame(
        {"y_pred": np.squeeze(nn_regressor.model.predict(x_test)), "y_true": y_test}
    )
    logger.info(compute_metric(result_test))
    return nn_regressor, result_test


def train_bagging(
    model_name,
    path,
    x_train,
    x_test,
    y_train,
    y_test,
    confs_nn,
    n_reti=500,
):

    for rete in range(n_reti):
        logger.info(f"Training network {rete}...")
        neural_net, _ = train(model_name, x_train, x_test, y_train, y_test, confs_nn)
        path_impianto = select_or_create(os.path.join(path, "models", f"{model_name}"))
        FILENAME = os.path.join(path_impianto, f"{rete}")
        neural_net.model.save(FILENAME)


def compute_metric(result: pd.DataFrame) -> float:
    """
    Function to compute the metric to evaluate the model.

    Parameters
    ----------
    result : pd.DataFrame
        dataframe with y_true / y_pred columns

    Returns
    -------
    float
        metric
    """
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

    path = os.getcwd()
    yaml_path = os.path.join(path, "models.yaml")
    confs_nn = load_yaml(yaml_path)

    n_reti = 50
    dataset = pd.read_csv("sample_dataset.csv")
    model_list = set(dataset["model_id"].values)

    for name in model_list:

        x = dataset.loc[dataset["model_id"] == name].drop(columns="target")
        y = dataset["target"].loc[dataset["model_id"] == name]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.1, shuffle=False
        )

        # TRAINING AND SAVING RESULTS
        # --------------------------------------------------------------------------------------------

        train_bagging(
            path=path,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            confs_nn=confs_nn,
            n_reti=n_reti,
        )

        logger.info(f"Train plant {name} completed!")

    logger.info("...MODEL FINISHED!!!")
