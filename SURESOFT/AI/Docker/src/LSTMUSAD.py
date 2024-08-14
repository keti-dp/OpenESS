# -*- coding: utf-8 -*-
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from trainer.LSTM_USAD_trainer import LSTMUsadModel

"""
Audibert, J., Michiardi, P., Guyard, F., Marti, S., Zuluaga, M. A. (2020).
USAD : UnSupervised Anomaly Detection on multivariate time series.
Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, August 23-27, 2020
"""


class LSTM_USAD_train:
    def __init__(
        self, batch_size: int, n_epochs: int, lr: float, hidden_size: int = 300
    ):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.hidden_size = hidden_size

        pass

    def fit_scaler(self, data):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        self.save_scaler(scaler)

        return data

    @classmethod
    def save_scaler(cls, scaler):
        joblib.dump(scaler, "./model/lstm_usad_scaler.pkl")
        print("USAD Scaler Saved to src/model/usad_scaler.pkl")
        pass

    def load_dataset(self, data, dim, features):

        current = data[
            np.arange(dim)[None, :] + np.arange(data.shape[0] - dim, step=dim)[:, None]
        ]

        if len(features) == 1:
            current = np.swapaxes(current, 1, 2)

        self.w_size = current.shape[2]
        self.z_size = current.shape[1] * self.hidden_size

        windows_normal_train = current[: int(np.floor(0.7 * current.shape[0]))]
        windows_normal_val = current[
            int(np.floor(0.7 * current.shape[0])) : int(np.floor(current.shape[0]))
        ]
        windows_normal_train = torch.from_numpy(windows_normal_train).float()
        windows_normal_val = torch.from_numpy(windows_normal_val).float()
        # self.w_size = dim * len(features)
        # self.z_size = dim * self.hidden_size

        train_loader = DataLoader(
            windows_normal_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        val_loader = DataLoader(
            windows_normal_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, val_loader

    def fit(self, train_loader, val_loader, lr, es_epochs):

        self.model = LSTMUsadModel(self.w_size, self.z_size)
        print(self.model)
        self.model = self.model.to_device(self.model, self.model.device)
        history = self.model.fit(self.n_epochs, train_loader, val_loader, lr, es_epochs)

        return history

    def save_model(self, FEATURES):

        torch.save(
            {
                "features": FEATURES,
                "encoder": self.model.encoder.state_dict(),
                "decoder1": self.model.decoder1.state_dict(),
                "decoder2": self.model.decoder2.state_dict(),
            },
            "./model/lstm_usad_model.pth",
        )
        print("LSTM_USAD Model Saved to model/lstm_usad_model.pth")


class LSTM_USAD_pred:
    def __init__(
        self,
        scaler_path: str = "./model/lstm_usad_scaler.pkl",
        model_path: str = "./model/lstm_usad_model.pth",
    ):
        self.__scaler = self.load_scaler(scaler_path)
        self.model, self._w_size, self._z_size = self.load_model(model_path)

    def load_scaler(self, scaler_path):
        scaler = joblib.load(scaler_path)
        return scaler

    def load_model(self, model_path):

        checkpoint = torch.load(model_path)

        w_size = checkpoint["encoder"]["lstm1.weight_ih_l0"].shape[1]
        z_size = checkpoint["encoder"]["hidden_to_latent.weight"].shape[0]
        model = LSTMUsadModel(w_size, z_size)
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.decoder1.load_state_dict(checkpoint["decoder1"])
        model.decoder2.load_state_dict(checkpoint["decoder2"])
        model = model.to_device(model, model.device)
        return model, w_size, z_size

    def load_dataset(self, data, features):
        time_index = data.index.values[
            np.arange(128, len(data), step=10).reshape(-1, 1)
        ]

        data = self.__scaler.transform(data)
        # data = data.reshape(-1,self._w_size)

        data = data[
            np.arange(128)[None, :] + np.arange(data.shape[0] - 128, step=10)[:, None]
        ]
        if len(features) == 1:
            data = np.swapaxes(data, 1, 2)
        data = torch.from_numpy(data).float()

        test_loader = DataLoader(
            data,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        return test_loader, time_index

    def anomaly_detection(self, test_loader):

        result = self.model.testing(test_loader)
        print("Predict Done")
        return result
