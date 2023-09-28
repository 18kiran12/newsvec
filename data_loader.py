# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load raw data and generate time series dataset."""

import os
import pdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from semantic_embedder import create_embeddings
import pickle


DATA_DIR = 'gs://time_series_datasets'
LOCAL_CACHE_DIR = './dataset/'


class TSFDataLoader:
  """Generate data loader from raw data."""

  def __init__(
      self, data, batch_size, seq_len, pred_len, feature_type, target='OT'
  ):
    self.data = data
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.feature_type = feature_type
    self.target = target
    self.target_slice = slice(0, None)

    self._read_data()

  def _read_data(self):
    """Load raw data and split datasets."""

    # copy data from cloud storage if not exists
    if not os.path.isdir(LOCAL_CACHE_DIR):
      os.mkdir(LOCAL_CACHE_DIR)

    file_name = self.data + '.csv'
    cache_filepath = os.path.join(LOCAL_CACHE_DIR, file_name)
    if not os.path.isfile(cache_filepath):
      tf.io.gfile.copy(
          os.path.join(DATA_DIR, file_name), cache_filepath, overwrite=True
      )

    df_raw = pd.read_csv(cache_filepath)

    # S: univariate-univariate, M: multivariate-multivariate, MS:
    # multivariate-univariate
    df = df_raw.set_index('date')
    df.drop(columns = ["cases"], inplace= True)
    if self.feature_type == 'S':
      df = df[[self.target]]
    elif self.feature_type == 'MS':
      target_idx = df.columns.get_loc(self.target)
      self.target_slice = slice(target_idx, target_idx + 1)

    # split train/valid/test
    n = len(df)
    if self.data.startswith('ETTm'):
      train_end = 12 * 30 * 24 * 4
      val_end = train_end + 4 * 30 * 24 * 4
      test_end = val_end + 4 * 30 * 24 * 4
    elif self.data.startswith('ETTh'):
      train_end = 12 * 30 * 24
      val_end = train_end + 4 * 30 * 24
      test_end = val_end + 4 * 30 * 24
    else:
      train_end = int(n * 0.7)
      val_end = n - int(n * 0.2)
      test_end = n
    train_df = df[:train_end]
    val_df = df[train_end - self.seq_len : val_end]
    test_df = df[val_end - self.seq_len : test_end]

    # standardize by training set
    self.scaler = StandardScaler()
    self.scaler.fit(train_df.values)

    def scale_df(df, scaler):
      data = scaler.transform(df.values)
      return pd.DataFrame(data, index=df.index, columns=df.columns)

    self.train_df = scale_df(train_df, self.scaler)
    self.val_df = scale_df(val_df, self.scaler)
    self.test_df = scale_df(test_df, self.scaler)
    self.n_feature = self.train_df.shape[-1]
    print(self.train_df.shape, self.val_df.shape, self.test_df.shape)
    nan_rows = self.train_df[self.train_df.isna().any(axis=1)]
    print("train nans : ", nan_rows.index)
    nan_rows = self.val_df[self.val_df.isna().any(axis=1)]
    print("val nans : ", nan_rows.index)
    nan_rows = self.test_df[self.test_df.isna().any(axis=1)]
    print("test nans : ", nan_rows.index)

    if self.feature_type == "MS":
      self.add_embeddings()

  def add_embeddings(self):
    # date_embeddings = create_embeddings("./dataset/COVID_India.json")
    # self.train_df['date'] = self.train_df['date'].dt.strftime('%d/%m/%Y')
    with open('date_embeddings.pickle', 'rb') as openfile:
      date_embeddings = pickle.load(openfile)
    new_embeddings = {}
    for key, value in date_embeddings.items():
      # pdb.set_trace()
      new_embeddings[key] = np.sum(value, axis = 0).tolist()
    # pdb.set_trace()
    
    embedded_df = pd.DataFrame.from_dict(new_embeddings).T
    embedded_df.index = pd.to_datetime(embedded_df.index) #+ pd.Timedelta(days=1)
    

    self.train_df.index = pd.to_datetime(self.train_df.index)
    self.val_df.index = pd.to_datetime(self.val_df.index)
    self.test_df.index = pd.to_datetime(self.test_df.index)

    print("Before ", self.train_df.shape, self.val_df.shape, self.test_df.shape)
    self.train_df = pd.merge(self.train_df, embedded_df, how="left", left_index=True, right_index=True)
    # self.train_df.fillna(method='ffill', inplace =True)
    self.val_df = pd.merge(self.val_df, embedded_df, how="left", left_index=True, right_index=True)
    # self.val_df.fillna(method='ffill', inplace =True)
    self.test_df = pd.merge(self.test_df, embedded_df, how="left", left_index=True, right_index=True)
    # self.test_df.fillna(method='ffill', inplace =True)
    # pdb.set_trace()
    # self.train_df = self.train_df.reset_index(level=0)
    # self.val_df = self.val_df.reset_index(level=0)
    # self.test_df = self.test_df.reset_index(level=0)

    nan_rows = self.train_df[self.train_df.isna().any(axis=1)]
    print("train nans : ", nan_rows.index)
    nan_rows = self.val_df[self.val_df.isna().any(axis=1)]
    print("val nans : ", nan_rows.index)
    nan_rows = self.test_df[self.test_df.isna().any(axis=1)]
    print("test nans : ", nan_rows.index)
    # self.train_df[self.train_df.index==pd.to_datetime('2020-01-02')]
    print("After ", self.train_df.shape, self.val_df.shape, self.test_df.shape)
    # date_embeddings
    # pdb.set_trace()
    

  def _split_window(self, data):
    inputs = data[:, : self.seq_len, :]
    labels = data[:, self.seq_len :, self.target_slice]
    # pdb.set_trace()
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.seq_len, None])
    labels.set_shape([None, self.pred_len, None])
    return inputs, labels

  def _make_dataset(self, data, shuffle=True):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=(self.seq_len + self.pred_len),
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=self.batch_size,
    )
    ds = ds.map(self._split_window)
    return ds

  def inverse_transform(self, data):
    return self.scaler.inverse_transform(data)

  def get_train(self, shuffle=True):
    return self._make_dataset(self.train_df, shuffle=shuffle)

  def get_val(self):
    return self._make_dataset(self.val_df, shuffle=False)

  def get_test(self):
    return self._make_dataset(self.test_df, shuffle=False)
