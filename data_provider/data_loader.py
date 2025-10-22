import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.HydrologicalScaler import HybridScaler
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


class Dataset_HybridScaler(Dataset):
    _processed_data_cache = None

    def __init__(self, root_path, flag='train', size=None, data_path='data.csv',
                 target='OT', timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        if Dataset_HybridScaler._processed_data_cache is not None:
            cache = Dataset_HybridScaler._processed_data_cache
            self.scaler = cache['scaler']
            self.rainfall_cols = cache['rainfall_cols']
            self.runoff_cols = cache['runoff_cols']
            all_rainfall_data = cache['rainfall_data']
            all_runoff_data = cache['runoff_data']
            all_data_stamp = cache['data_stamp']
            date_series = cache.get('date_series', None)
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]

            cols_Q = df_raw.columns[df_raw.columns.str.contains('Q', na=False)]
            cols_drp = df_raw.columns[df_raw.columns.str.contains('DRP', na=False)]

            self.scaler = HybridScaler(cols_drp, cols_Q)
            self.rainfall_cols = list(cols_drp)
            self.runoff_cols = list(cols_Q)

            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + int(len(df_raw) * 0.1), len(df_raw)]

            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]

            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values, df_data.columns)

            all_rainfall_data, all_runoff_data = self.scaler.transform(df_data.values)

            df_stamp_all = df_raw[['date']].copy()
            df_stamp_all['date'] = pd.to_datetime(df_stamp_all.date)

            date_series = df_stamp_all['date']

            if self.timeenc == 0:
                df_stamp_all['month'] = df_stamp_all.date.apply(lambda row: row.month, 1)
                df_stamp_all['day'] = df_stamp_all.date.apply(lambda row: row.day, 1)
                df_stamp_all['weekday'] = df_stamp_all.date.apply(lambda row: row.weekday(), 1)
                df_stamp_all['hour'] = df_stamp_all.date.apply(lambda row: row.hour, 1)
                all_data_stamp = df_stamp_all.drop(columns=['date']).values
            elif self.timeenc == 1:
                all_data_stamp = time_features(pd.to_datetime(df_stamp_all['date'].values), freq=self.freq)
                all_data_stamp = all_data_stamp.transpose(1, 0)

            Dataset_HybridScaler._processed_data_cache = {
                'scaler': self.scaler,
                'rainfall_cols': self.rainfall_cols,
                'runoff_cols': self.runoff_cols,
                'rainfall_data': all_rainfall_data,
                'runoff_data': all_runoff_data,
                'data_stamp': all_data_stamp,
                'date_series': date_series,
            }

        num_train = int(len(all_runoff_data) * 0.7)
        num_test = int(len(all_runoff_data) * 0.2)
        num_vali = len(all_runoff_data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(all_runoff_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(all_runoff_data)]

        if self.set_type == 0:
            self.border1 = border1s[0]
            self.border2 = border2s[0]
        elif self.set_type == 1:
            self.border1 = border1s[1]
            self.border2 = border2s[1]
        else:
            self.border1 = border1s[2]
            self.border2 = border2s[2]

        self.data_x_rain = all_rainfall_data[self.border1:self.border2]
        self.data_x_runoff = all_runoff_data[self.border1:self.border2]
        self.data_stamp = all_data_stamp[self.border1:self.border2]

        self._date_slice = None
        if isinstance(date_series, (pd.Series, pd.Index)):
            self._date_slice = date_series.iloc[self.border1:self.border2]

        if self.target in self.runoff_cols:
            self.target_index = self.runoff_cols.index(self.target)
            self.is_runoff_target = True
            self.data_y = all_runoff_data[self.border1:self.border2, self.target_index].reshape(-1, 1)
        elif self.target in self.rainfall_cols:
            self.target_index = self.rainfall_cols.index(self.target)
            self.is_runoff_target = False
            self.data_y = all_rainfall_data[self.border1:self.border2, self.target_index].reshape(-1, 1)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        rain = self.data_x_rain[s_begin:s_end]
        runoff = self.data_x_runoff[s_begin:s_end]
        target = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        input_list = [runoff, rain]
        return input_list, target, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x_runoff) - self.seq_len - self.pred_len + 1

    def get_rainfall_cols(self):
        return self.rainfall_cols

    def get_runoff_cols(self):
        return self.runoff_cols

    def inverse_transform(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if self.is_runoff_target:
            return self.scaler.inverse_transform_target(data, self.target_index)
        else:
            return self.scaler.inverse_transform_rainfall(data)

    def inverse_transform_rainfall(self, data):
        return self.scaler.inverse_transform_rainfall(data)

    def inverse_transform_runoff(self, data):
        return self.scaler.inverse_transform_runoff(data)

    def calculate_weights(self, rainfall_weight=0.4, target_weight=0.6, max_weight=10.0):
        weights = []
        for i in range(len(self.data_y) - self.pred_len + 1):
            target_sample = self.data_y[i:i + self.pred_len]
            target_mean = np.mean(target_sample)
            if target_mean < 0:
                target_mean = 0
            target_component = np.exp(target_mean)

            s_begin = i
            s_end = s_begin + self.seq_len
            seq_rainfall = self.data_x_rain[s_begin:s_end]
            total_rainfall_by_time = np.sum(np.abs(seq_rainfall), axis=1)
            if len(total_rainfall_by_time) > 0:
                max_total_rainfall = np.max(total_rainfall_by_time)
                sum_total_rainfall = np.sum(total_rainfall_by_time)
                rain_intensity = sum_total_rainfall / len(total_rainfall_by_time)
                rainfall_component = np.exp(max_total_rainfall * 0.5 + rain_intensity * 0.5)
            else:
                rainfall_component = 1.0
            w = target_weight * target_component + rainfall_weight * rainfall_component
            w = min(w, max_weight)
            weights.append(w)

        weights = np.array(weights)
        weights = np.nan_to_num(weights)
        weights = np.maximum(weights, 0)
        if np.sum(weights) > 0:
            return weights / np.sum(weights)
        else:
            return np.ones_like(weights) / len(weights)

    def get_pred_start_timestamp(self, sample_index: int):
        if self._date_slice is None:
            return None
        if sample_index < 0 or sample_index >= len(self):
            return None
        global_pred_start = self.border1 + sample_index + self.seq_len
        if global_pred_start + self.pred_len > self.border2:
            return None
        local_idx = global_pred_start - self.border1
        if local_idx < 0 or local_idx >= len(self._date_slice):
            return None
        return self._date_slice.iloc[local_idx]

class DatasetUnifiedStd(Dataset):
    _processed_data_cache = None
    def __init__(self, root_path, flag='train', size=None, data_path='data.csv',
                 target='OT', timeenc=0, freq='h'):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        if DatasetUnifiedStd._processed_data_cache is not None:
            cache = DatasetUnifiedStd._processed_data_cache
            self.scaler = cache['scaler']
            self.rainfall_cols = cache['rainfall_cols']
            self.runoff_cols = cache['runoff_cols']
            self.feature_cols = cache['feature_cols']
            all_rainfall_data = cache['rainfall_data']
            all_runoff_data = cache['runoff_data']
            all_data_stamp = cache['data_stamp']
            date_series = cache.get('date_series', None)
            feature_means = cache['feature_means']
            feature_scales = cache['feature_scales']
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]

            cols_Q = df_raw.columns[df_raw.columns.str.contains('Q', na=False)]
            cols_drp = df_raw.columns[df_raw.columns.str.contains('DRP', na=False)]

            self.rainfall_cols = list(cols_drp)
            self.runoff_cols = list(cols_Q)

            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + int(len(df_raw) * 0.1), len(df_raw)]

            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]


            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler = StandardScaler()
            self.scaler.fit(train_data.values)

            data_scaled = self.scaler.transform(df_data.values).astype(np.float32)
            df_scaled = pd.DataFrame(data_scaled, index=df_data.index, columns=df_data.columns)

            self.feature_cols = list(df_scaled.columns)
            feature_means = pd.Series(self.scaler.mean_, index=self.feature_cols)
            feature_scales = pd.Series(self.scaler.scale_, index=self.feature_cols)

            all_rainfall_data = df_scaled[self.rainfall_cols].values if self.rainfall_cols else np.zeros((len(df_scaled), 0), dtype=np.float32)
            all_runoff_data = df_scaled[self.runoff_cols].values if self.runoff_cols else np.zeros((len(df_scaled), 0), dtype=np.float32)

            df_stamp_all = df_raw[['date']].copy()
            df_stamp_all['date'] = pd.to_datetime(df_stamp_all.date)
            date_series = df_stamp_all['date']

            if self.timeenc == 0:
                df_stamp_all['month'] = df_stamp_all.date.apply(lambda row: row.month, 1)
                df_stamp_all['day'] = df_stamp_all.date.apply(lambda row: row.day, 1)
                df_stamp_all['weekday'] = df_stamp_all.date.apply(lambda row: row.weekday(), 1)
                df_stamp_all['hour'] = df_stamp_all.date.apply(lambda row: row.hour, 1)
                all_data_stamp = df_stamp_all.drop(columns=['date']).values
            elif self.timeenc == 1:
                all_data_stamp = time_features(pd.to_datetime(df_stamp_all['date'].values), freq=self.freq)
                all_data_stamp = all_data_stamp.transpose(1, 0)

            DatasetUnifiedStd._processed_data_cache = {
                'scaler': self.scaler,
                'rainfall_cols': self.rainfall_cols,
                'runoff_cols': self.runoff_cols,
                'feature_cols': self.feature_cols,
                'rainfall_data': all_rainfall_data,
                'runoff_data': all_runoff_data,
                'data_stamp': all_data_stamp,
                'date_series': date_series,
                'feature_means': feature_means,
                'feature_scales': feature_scales,
            }

        self._feature_means = feature_means
        self._feature_scales = feature_scales

        num_train = int(len(all_runoff_data) * 0.7)
        num_test = int(len(all_runoff_data) * 0.2)
        num_vali = len(all_runoff_data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(all_runoff_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(all_runoff_data)]

        if self.set_type == 0:
            self.border1 = border1s[0]; self.border2 = border2s[0]
        elif self.set_type == 1:
            self.border1 = border1s[1]; self.border2 = border2s[1]
        else:
            self.border1 = border1s[2]; self.border2 = border2s[2]

        self.data_x_rain = all_rainfall_data[self.border1:self.border2]
        self.data_x_runoff = all_runoff_data[self.border1:self.border2]
        self.data_stamp = all_data_stamp[self.border1:self.border2]

        self._date_slice = None
        if isinstance(date_series, (pd.Series, pd.Index)):
            self._date_slice = date_series.iloc[self.border1:self.border2]

        if self.target in self.runoff_cols:
            self.target_index = self.runoff_cols.index(self.target)
            self.is_runoff_target = True
            self.data_y = self.data_x_runoff[:, self.target_index].reshape(-1, 1)
        elif self.target in self.rainfall_cols:
            self.target_index = self.rainfall_cols.index(self.target)
            self.is_runoff_target = False
            self.data_y = self.data_x_rain[:, self.target_index].reshape(-1, 1)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        rain = self.data_x_rain[s_begin:s_end]
        runoff = self.data_x_runoff[s_begin:s_end]
        target = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        input_list = [runoff, rain]
        return input_list, target, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x_runoff) - self.seq_len - self.pred_len + 1

    def get_rainfall_cols(self):
        return self.rainfall_cols

    def get_runoff_cols(self):
        return self.runoff_cols

    def _inverse_by_cols(self, data, cols):
        arr = np.asarray(data, dtype=np.float32)
        orig_shape = arr.shape
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        mu = np.array([self._feature_means[c] for c in cols], dtype=np.float32)
        sigma = np.array([self._feature_scales[c] for c in cols], dtype=np.float32)

        arr = arr.reshape(-1, len(cols))
        inv = arr * sigma.reshape(1, -1) + mu.reshape(1, -1)
        return inv.reshape(orig_shape)

    def inverse_transform(self, data):
        if self.is_runoff_target:
            col = self.runoff_cols[self.target_index]
        else:
            col = self.rainfall_cols[self.target_index]
        return self._inverse_by_cols(data, [col])

    def inverse_transform_rainfall(self, data):
        if not self.rainfall_cols:
            return np.asarray(data)
        return self._inverse_by_cols(data, self.rainfall_cols)

    def inverse_transform_runoff(self, data):
        if not self.runoff_cols:
            return np.asarray(data)
        return self._inverse_by_cols(data, self.runoff_cols)

    def get_pred_start_timestamp(self, sample_index: int):
        if self._date_slice is None:
            return None
        if sample_index < 0 or sample_index >= len(self):
            return None
        global_pred_start = self.border1 + sample_index + self.seq_len
        if global_pred_start + self.pred_len > self.border2:
            return None
        local_idx = global_pred_start - self.border1
        if local_idx < 0 or local_idx >= len(self._date_slice):
            return None
        return self._date_slice.iloc[local_idx]
