import pandas as pd
import numpy as np
from typing import Optional


def get_preprocessing_function(
        df: pd.DataFrame,
        relative_change: bool = True,
        relative_to: bool = True
):
    columns = []

    if relative_change:
        def get_relative_change(value):
            return 100 * (value.shift(1) - 1) / value
        df['r_open'] = get_relative_change(df['open'])
        df['r_close'] = get_relative_change(df['close'])
        df['r_high'] = get_relative_change(df['high'])
        df['r_low'] = get_relative_change(df['low'])
        r_columns = ['r_open', 'r_close', 'r_high', 'r_low']
        columns += r_columns

    if relative_to:
        def get_relative_to(value, reference):
            return 100 * (value - 1) / reference
        df['rto_close'] = get_relative_to(df['close'], df['open'])
        df['rto_low'] = get_relative_to(df['low'], df['open'])
        df['rto_high'] = get_relative_to(df['high'], df['open'])
        rto_columns = ['rto_close', 'rto_low', 'rto_high']
        columns += rto_columns

    return df[columns]


def get_pca_preprocessing_function(
        epsilon: float = 10e-9,
        pca_num_componants: int = 5,
        means: Optional[list] = None,  # todo maybe use config file instead ?
        stds: Optional[list] = None,
        pca_components: Optional[list] = None
):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    def preprocess(dataframe):
        dataframe[['open_s', 'low_s', 'close_s', 'high_s']] = dataframe[['open', 'low', 'close', 'high']].shift(1)
        nominator = np.array(dataframe[['open_s', 'low_s', 'close_s', 'high_s', 'open', 'low', 'close', 'high']])
        denominator = np.array(dataframe[['open', 'low', 'close', 'high']])

        nominator = np.nan_to_num(nominator, nan=0.0)
        denominator = np.nan_to_num(denominator, nan=0.0)

        nominator = nominator + epsilon
        denominator = denominator + epsilon

        nominator_ = np.reshape(np.repeat(nominator, 4), (-1, 8, 4))
        denominator_ = np.tile(np.expand_dims(denominator, axis=1), (1, 8, 1))
        new = nominator_/denominator_

        features = new.reshape(new.shape[0], -1)

        scaler = StandardScaler()  # todo find what to do with the scaler, mean std all data, put in config file
        scaler.fit(features)
        scaled_features = scaler.transform(features)

        pca = PCA(n_components=pca_num_componants)  # todo idem for PCA
        pca.fit(scaled_features)

        data = pca.transform(scaled_features)
        columns = [f'pca_{i}' for i in range(pca_num_componants)]
        return pd.DataFrame(data, columns=columns)

    return preprocess
