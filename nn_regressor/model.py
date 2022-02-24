# -*- coding: utf-8 -*-
import logging
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers.experimental import preprocessing

logger = logging.getLogger("__main__")


def step_decay(epoch: int) -> float:
    """
    _summary_

    Parameters
    ----------
    epoch : int
        _description_

    Returns
    -------
    float
        _description_
    """
    initial_lrate = 0.001
    drop = 0.00025
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


class NeuralNetworkRegressor:
    """
    _summary_
    """

    def __init__(self, conf: dict):
        """
        _summary_

        Parameters
        ----------
        conf : dict
            _description_
        """
        self.parameters = conf

        self.loss = conf["loss"]
        self.epochs = 1000
        self.early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        self.decay_lr = LearningRateScheduler(step_decay)
        self.optimizer = conf["optimizer"]
        self.first_layer = conf["first_layer"]
        self.hidden_layers = conf["hidden_layers"]
        self.output_layer = conf["output_layer"]

        self.model = None
        self.metric = None

    @staticmethod
    def get_layer(layer_settings: dict, layerName: str):
        """
        _summary_

        Parameters
        ----------
        layer_settings : dict
            _description_
        layerName : str
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if "dense" in layerName:
            return layers.Dense(**layer_settings, name=layerName)
        elif "dropout" in layerName:
            return layers.Dropout(**layer_settings, name=layerName)
        else:
            raise ValueError

    def build_architecture(self, n_features, normalizer):
        prep_layer = [normalizer]
        input_layer = [
            layers.Dense(
                **self.first_layer, name="input_layer", input_shape=[n_features]
            )
        ]
        hidden_layers = [
            self.get_layer(myLayer, myName)
            for myName, myLayer in zip(
                self.hidden_layers.keys(), self.hidden_layers.values()
            )
        ]
        output_layer = [layers.Dense(**self.output_layer, name="output_layer")]
        return prep_layer + input_layer + hidden_layers + output_layer

    def build_model(self, n_features: int, normalizer) -> tf.keras.Model:
        architecture = self.build_architecture(n_features, normalizer)
        model = keras.Sequential(architecture)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.loss])
        return model

    def fit(self, x, y, **args) -> tf.keras.callbacks.History:
        """
        Fit the deep learning model and creating the normalizer.

        Parameters
        ----------
        x : _type_
            _description_
        y : _type_
            _description_

        Returns
        -------
        tf.keras.callbacks.History
            _description_
        """
        normalizer = preprocessing.Normalization(name="normalization")
        normalizer.adapt(x)
        self.model = self.model or self.build_model(x.shape[1], normalizer=normalizer)
        return self.model.fit(
            x,
            y,
            epochs=self.epochs,
            validation_split=0.1,
            callbacks=[self.early_stop, self.decay_lr],
            **args,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def get_built_model(self, x):
        normalizer = preprocessing.Normalization(name="normalization")
        mod_predict = self.build_model(x.shape[1], normalizer=normalizer)
        mod_predict.build((None, x.shape[1]))
        return mod_predict

    def plot_history(
        self, history: tf.keras.callbacks.History, title="", full_filename=None
    ) -> None:
        """
        Plot training vs validation loss over epoch.
        """
        import matplotlib.pyplot as plt

        hist = pd.DataFrame(history.history)
        hist["epoch"] = history.epoch

        plt.figure()
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(f"{self.loss} over epoch")
        plt.plot(hist["epoch"], hist[self.loss], label="Train Error")
        plt.plot(hist["epoch"], hist[f"val_{self.loss}"], label="Val Error")
        plt.legend()
        if full_filename is not None:
            plt.savefig(full_filename, dpi=300)
        plt.show(block=False)
        return None

    def summary(self) -> None:
        return self.model.summary()
