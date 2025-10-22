from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class SparseScaler:
    def __init__(self):
        pass

    def fit(self, X):

        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            device = X.device
            X_np = X.cpu().numpy()
        else:
            device = None
            X_np = X

        mask = X_np != 0.0
        result = np.zeros_like(X_np)

        if np.any(mask):
            non_zero = X_np[mask]
            transformed = np.log1p(non_zero)
            result[mask] = transformed

        if device is not None:
            result = torch.from_numpy(result).to(device)

        return result

    def inverse_transform(self, X):
        if isinstance(X, torch.Tensor):
            device = X.device
            X_np = X.cpu().numpy()
        else:
            device = None
            X_np = X

        mask = X_np != 0.0
        result = np.zeros_like(X_np)

        if np.any(mask):
            transformed = X_np[mask]
            original = np.expm1(transformed)
            result[mask] = original

        if device is not None:
            result = torch.from_numpy(result).to(device)

        return result

class HybridScaler(BaseEstimator, TransformerMixin):
    def __init__(self, rainfall_cols: List[str], runoff_cols: List[str]):
        self.rainfall_cols = rainfall_cols
        self.runoff_cols = runoff_cols
        self.rainfall_scaler = SparseScaler()
        self.runoff_scaler = StandardScaler()

    def fit(self, X: np.ndarray, columns: Union[List[str], pd.Index], y=None):
        self.columns = columns

        columns_list = list(columns) if isinstance(columns, pd.Index) else columns

        self.rainfall_indices = []
        for col in self.rainfall_cols:
            if col in columns_list:
                if isinstance(columns, pd.Index):
                    self.rainfall_indices.append(columns.get_loc(col))
                else:
                    self.rainfall_indices.append(columns_list.index(col))

        self.runoff_indices = []
        for col in self.runoff_cols:
            if col in columns_list:
                if isinstance(columns, pd.Index):
                    self.runoff_indices.append(columns.get_loc(col))
                else:
                    self.runoff_indices.append(columns_list.index(col))

        if len(self.rainfall_indices) > 0:
            self.rainfall_scaler.fit(X[:, self.rainfall_indices])

        if len(self.runoff_indices) > 0:
            self.runoff_scaler.fit(X[:, self.runoff_indices])

        return self

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rainfall_data = X[:, self.rainfall_indices].copy() if self.rainfall_indices else np.array([]).reshape(
            X.shape[0], 0)
        runoff_data = X[:, self.runoff_indices].copy() if self.runoff_indices else np.array([]).reshape(X.shape[0], 0)

        if rainfall_data.size > 0:
            rainfall_data = self.rainfall_scaler.transform(rainfall_data)

        if len(self.runoff_indices) > 0:
            runoff_data = self.runoff_scaler.transform(runoff_data)

        return rainfall_data, runoff_data

    def inverse_transform_rainfall(self, rainfall_data: np.ndarray) -> np.ndarray:
        if rainfall_data.size == 0:
            return rainfall_data

        rainfall_original = self.rainfall_scaler.inverse_transform(rainfall_data)
        rainfall_original = np.maximum(rainfall_original, 0)
        return rainfall_original

    def inverse_transform_runoff(self, runoff_data: np.ndarray) -> np.ndarray:
        if runoff_data.size == 0:
            return runoff_data

        return self.runoff_scaler.inverse_transform(runoff_data)

    def get_target_scaler(self, target_col: str):
        target_index = list(self.runoff_cols).index(target_col)
        return target_index

    def inverse_transform_target(self, y: np.ndarray, target_index: int) -> np.ndarray:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[1] == 1:
            temp = np.zeros((y.shape[0], len(self.runoff_cols)))
            temp[:, target_index] = y.flatten()

            temp = self.runoff_scaler.inverse_transform(temp)

            return temp[:, target_index].reshape(y.shape)
        else:
            return self.runoff_scaler.inverse_transform(y)
