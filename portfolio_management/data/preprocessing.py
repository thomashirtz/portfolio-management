from typing import Dict
from typing import Optional
from typing import Tuple

from abc import ABC
from abc import abstractmethod

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Preprocessing(ABC):
    @abstractmethod
    def __call__(
            self,
            dataframe_dict: Dict[str, pd.DataFrame],
            is_test: bool
    ) -> Dict[str, pd.DataFrame]:
        pass


class PCAPreprocessing(Preprocessing):
    def __init__(
            self,
            epsilon: float = 10e-9,
            pca_num_components: int = 5,
    ):
        super(PCAPreprocessing, self).__init__()
        self.epsilon = epsilon
        self.pca_num_components = pca_num_components
        self.is_trained = False

        self.pca = None
        self.scaler_dict = {}

    def __call__(
            self,
            dataframe_dict: Dict[str, pd.DataFrame],
            is_test: bool
    ) -> Dict[str, pd.DataFrame]:

        if is_test and not self.is_trained:
            raise ValueError

        features_scaled_dict = {}
        for symbol, dataframe in dataframe_dict.items():
            features = self.get_features(dataframe)
            features_scaled, scaler = self.scale(features, symbol)
            features_scaled_dict[symbol] = features_scaled
            if not is_test:
                self.scaler_dict[symbol] = scaler

        preprocessed_dataframe_dict = self.apply_pca(features_scaled_dict, is_test)

        if not is_test:
            self.is_trained = True

        return preprocessed_dataframe_dict

    def scale(self, features: np.array, symbol: str) -> Tuple[np.array, StandardScaler]:
        if symbol not in self.scaler_dict.keys():
            scaler = StandardScaler()
            scaler.fit(features)
        else:
            scaler = self.scaler_dict[symbol]

        scaled_features = scaler.transform(features)
        return scaled_features, scaler

    def apply_pca(self, features_dict: Dict[str, pd.DataFrame], is_test: bool) -> Dict[str, pd.DataFrame]:
        if not is_test:
            self.pca = PCA(n_components=self.pca_num_components)
            features = np.concatenate(list(features_dict.values()))
            self.pca.fit(features)

        dataframe_dict = {}
        columns = [f'pca_{i}' for i in range(self.pca_num_components)]
        for symbol, features in features_dict.items():
            data = self.pca.transform(features)
            dataframe = pd.DataFrame(data, columns=columns)
            dataframe_dict[symbol] = dataframe

        return dataframe_dict

    def get_features(self, dataframe):
        dataframe[['open_s', 'low_s', 'close_s', 'high_s']] = dataframe[['open', 'low', 'close', 'high']].shift(-1)

        nominator = np.array(dataframe[['open_s', 'low_s', 'close_s', 'high_s', 'open', 'low', 'close', 'high']])
        denominator = np.array(dataframe[['open', 'low', 'close', 'high']])

        nominator = np.nan_to_num(nominator, nan=0.0)
        denominator = np.nan_to_num(denominator, nan=0.0)

        nominator = nominator + self.epsilon
        denominator = denominator + self.epsilon

        nominator_reshaped = np.reshape(np.repeat(nominator, 4), (-1, 8, 4))
        denominator_reshaped = np.tile(np.expand_dims(denominator, axis=1), (1, 8, 1))

        raw_features = nominator_reshaped / denominator_reshaped
        features = raw_features.reshape(raw_features.shape[0], -1)
        return features


def get_pca_preprocessing_function(
        epsilon: float = 10e-9,
        pca_num_components: int = 5,
        pca: Optional[PCA] = None,
        scaler: Optional[StandardScaler] = None,
):

    def preprocess(dataframe):
        dataframe[['open', 'low', 'close', 'high']] = dataframe[['open', 'low', 'close', 'high']].shift(1)
        dataframe[['open_s', 'low_s', 'close_s', 'high_s']] = dataframe[['open', 'low', 'close', 'high']].shift(1)

        nominator = np.array(dataframe[['open_s', 'low_s', 'close_s', 'high_s', 'open', 'low', 'close', 'high']])
        denominator = np.array(dataframe[['open', 'low', 'close', 'high']])

        nominator = np.nan_to_num(nominator, nan=0.0)
        denominator = np.nan_to_num(denominator, nan=0.0)

        nominator = nominator + epsilon
        denominator = denominator + epsilon

        nominator_reshaped = np.reshape(np.repeat(nominator, 4), (-1, 8, 4))
        denominator_reshaped = np.tile(np.expand_dims(denominator, axis=1), (1, 8, 1))

        raw_features = nominator_reshaped/denominator_reshaped
        features = raw_features.reshape(raw_features.shape[0], -1)

        nonlocal scaler
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(features)
        scaled_features = scaler.transform(features)

        nonlocal pca
        if pca is None:
            pca = PCA(n_components=pca_num_components)
            pca.fit(scaled_features)
        data = pca.transform(scaled_features)

        columns = [f'pca_{i}' for i in range(pca_num_components)]
        return pd.DataFrame(data, columns=columns)

    return preprocess
