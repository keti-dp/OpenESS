
# Code based on: https://github.com/datamllab/tods/tree/benchmark/benchmark
# Edited by: Eunseok Yang

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    # timestamp = np.linspace(0, 10, length)
    timestamp = np.arange(length)
    value = np.sin(2 * np.pi * freq * timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value

class UnivariateDataGenerator:
    def __init__(self):
        self.STREAM_LENGTH = None
        self.behavior_config = dict()

        self.data = None
        self.label = None
        self.data_origin = None
        self.timestamp = None

    def generate_timeseries(self, stream_length, behavior=sine, behavior_config=dict()):
        self.STREAM_LENGTH = stream_length
        self.behavior = behavior
        self.behavior_config = behavior_config

        self.behavior_config['length'] = self.STREAM_LENGTH
        self.data = self.behavior(**self.behavior_config)
        self.data_origin = self.data.copy()
        self.timestamp = np.arange(self.STREAM_LENGTH)

        self.label = np.zeros(self.STREAM_LENGTH, dtype=int)
        self.colored_label = np.ones(self.STREAM_LENGTH, dtype=int)

    def load_timeseries(self, ts):
        self.data = ts
        self.STREAM_LENGTH = len(ts)
        self.data_origin = self.data.copy()
        self.timestamp = np.arange(self.STREAM_LENGTH)

        self.label = np.zeros(self.STREAM_LENGTH, dtype=int)
        self.colored_label = np.ones(self.STREAM_LENGTH, dtype=int)
        

    # Changed code following the original paper
    # X_t = \mu(X) +- \lambda*\sigma(X)
    # factor noise is added
    def point_global_outliers(self, ratio, factor, factor_noise_amp):
        """
        Add point global outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: the larger, the outliers are farther from inliers
            factor_noise_amp: the noise to the factor
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio)) * self.STREAM_LENGTH).astype(int)
        # random noise to factors
        factors = np.random.randn(len(position)) * factor_noise_amp + factor
        maximum, minimum = max(self.data), min(self.data)

        for idx, i in enumerate(position):
            global_mean = self.data_origin.mean()
            global_std = self.data_origin.std()
            self.data[i] = global_mean + self.data_origin[i] * factors[idx] * global_std
            self.label[i] = 1
            self.colored_label[i] *= 2

    # Changed code following the original paper
    # X_t = \mu(X_(t-k, t+k)) +- \lambda*\sigma(X_(t-k, t+k))
    # factor noise is added
    def point_contextual_outliers(self, ratio, factor, radius, factor_noise_amp):
        """
        Add point contextual outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: the larger, the outliers are farther from inliers
            radius: the radius of collective outliers range
            factor_noise_amp: the noise to the factor
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio)) * self.STREAM_LENGTH).astype(int)
        # random noise to factors
        factors = np.random.randn(len(position)) * factor_noise_amp + factor
        maximum, minimum = max(self.data), min(self.data)

        for idx, i in enumerate(position):
            local_mean = self.data_origin[max(0, i - radius):min(i + radius, self.STREAM_LENGTH)].mean()
            local_std = self.data_origin[max(0, i - radius):min(i + radius, self.STREAM_LENGTH)].std()
            self.data[i] = local_mean + self.data_origin[i] * factors[idx] * local_std

            self.label[i] = 1
            self.colored_label[i] *= 3

    def collective_shapelet_outliers(self, ratio, radius, coef=3., noise_amp=0.0, base=[0.,]):
        """
        Add collective shapelet outliers to original data
        Args:
            ratio: what ratio outliers will be added
            radius: the radius of collective outliers range
            level: how many sine waves will square_wave synthesis
            base: a list of values that we want to substitute inliers when we generate outliers
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio / (2 * radius))) * self.STREAM_LENGTH).astype(int)
        
        value = np.resize(base / np.linalg.norm(base), self.STREAM_LENGTH)
        noise = np.random.normal(0, 1, self.STREAM_LENGTH)
        value = coef * value + noise_amp * noise
        sub_data = value

        for i in position:
            start, end = max(0, i - radius), min(self.STREAM_LENGTH, i + radius)
            self.data[start:end] += sub_data[start:end]
            self.label[start:end] = 1
            self.colored_label[start:end] *= 5

    def collective_trend_outliers(self, ratio, factor, radius):
        """
        Add collective trend outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: how dramatic will the trend be
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio / (2 * radius))) * self.STREAM_LENGTH).astype(int)
        for i in position:
            start, end = max(0, i - radius), min(self.STREAM_LENGTH, i + radius)
            slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
            self.data[start:end] = self.data_origin[start:end] + slope
            self.data[end:] = self.data[end:] + slope[-1]
            self.label[start:end] = 1
            self.colored_label[start:end] *= 7

    # This needs to be edited since we do not know the explicit frequency of realworld data.
    def collective_seasonal_outliers(self, ratio, factor, radius):
        """
        Add collective seasonal outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: how many times will frequency multiple
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio / (2 * radius))) * self.STREAM_LENGTH).astype(int)
        seasonal_config = self.behavior_config
        seasonal_config['freq'] = factor * self.behavior_config['freq']
        for i in position:
            start, end = max(0, i - radius), min(self.STREAM_LENGTH, i + radius)
            self.data[start:end] = self.behavior(**seasonal_config)[start:end]
            self.label[start:end] = 1
            self.colored_label[start:end] *= 11

    # new plot presenting each outlier
    def plot(self):
        colors = ['r', 'g', 'y', 'c', 'm']

        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        ax[0].plot(self.timestamp, self.data_origin)
        ax[1].plot(self.timestamp, self.data)

        for i, prime in enumerate([2, 3, 5, 7, 11]):
            outlier = [j for j, label in enumerate(self.colored_label) if label%prime == 0]
            ax[1].scatter(outlier, self.data[outlier], color=colors[i])

        plt.legend([
            'data', 'point global outlier', 'point contextual outlier', 'collective shaplet outlier', 
            'collective trend outlier', 'collective seasonal outlier'
            ], loc='lower right')

        ax[0].set_title('Original data')
        ax[1].set_title('Synthetic data')

        plt.savefig('./sample_figure.png')
        plt.show()

if __name__ == '__main__':
    np.random.seed(100)

    BEHAVIOR_CONFIG = {'freq': 0.04, 'coef': 1.5, "offset": 0.0, 'noise_amp': 0.05}
    BASE = [0, 1]

    univariate_data = UnivariateDataGenerator()
    univariate_data.generate_timeseries(stream_length=400, behavior=sine, behavior_config=BEHAVIOR_CONFIG)
    
    univariate_data.collective_shapelet_outliers(ratio=0.05, radius=5, coef=1.5, noise_amp=0.03, base=BASE) #2
    univariate_data.collective_seasonal_outliers(ratio=0.05, factor=3, radius=5) #3
    univariate_data.collective_trend_outliers(ratio=0.05, factor=0.5, radius=5) #4

    univariate_data.point_global_outliers(ratio=0.05, factor=3.5, factor_noise_amp=0.1) #0
    univariate_data.point_contextual_outliers(ratio=0.05, factor=2.5, radius=5, factor_noise_amp=0.1) #1
    
    univariate_data.plot()

    df = pd.DataFrame({'value': univariate_data.data, 'anomaly': univariate_data.label})
    df.to_csv('sample.csv', index=False)