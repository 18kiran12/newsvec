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

"""Train and evaluate models for time series forecasting."""

import argparse
import glob
import logging
import os
import time
import pdb
from data_loader import TSFDataLoader
import models
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def parse_args():
  """Parse the arguments for experiment configuration."""

  parser = argparse.ArgumentParser(
      description='TSMixer for Time Series Forecasting'
  )

  # basic config
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  parser.add_argument('--exp', type=str, default="trial", help='experiment_name')
  parser.add_argument(
      '--model',
      type=str,
      default='tsmixer_rev_in',
      help='model name, options: [tsmixer, tsmixer_rev_in]',
  )

  # data loader
  parser.add_argument(
      '--data',
      type=str,
      default='COVID_India',
      choices=[
          'electricity',
          'exchange_rate',
          'national_illness',
          'traffic',
          'weather',
          'ETTm1',
          'ETTm2',
          'ETTh1',
          'ETTh2',
          'COVID_India',
          'bundestag_weekly'
      ],
      help='data name',
  )
  parser.add_argument(
      '--feature_type',
      type=str,
      default='MS',
      choices=['S', 'M', 'MS'],
      help=(
          'forecasting task, options:[M, S, MS]; M:multivariate predict'
          ' multivariate, S:univariate predict univariate, MS:multivariate'
          ' predict univariate'
      ),
  )
  parser.add_argument(
      '--target', type=str, default='deaths', help='target feature in S or MS task'
  )
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='./checkpoints/',
      help='location of model checkpoints',
  )
  parser.add_argument(
      '--delete_checkpoint',
      action='store_true',
      help='delete checkpoints after the experiment',
  )

  # forecasting task
  parser.add_argument(
      '--seq_len', type=int, default=50, help='input sequence length'
  )
  parser.add_argument(
      '--pred_len', type=int, default=10, help='prediction sequence length'
  )

  # model hyperparameter
  parser.add_argument(
      '--n_block',
      type=int,
      default=2,
      help='number of block for deep architecture',
  )
  parser.add_argument(
      '--ff_dim',
      type=int,
      default=2048,
      help='fully-connected feature dimension',
  )
  parser.add_argument(
      '--dropout', type=float, default=0.05, help='dropout rate'
  )
  parser.add_argument(
      '--norm_type',
      type=str,
      default='B',
      choices=['L', 'B'],
      help='LayerNorm or BatchNorm',
  )
  parser.add_argument(
      '--activation',
      type=str,
      default='relu',
      choices=['relu', 'gelu'],
      help='Activation function',
  )
  parser.add_argument(
      '--kernel_size', type=int, default=4, help='kernel size for CNN'
  )
  parser.add_argument(
      '--temporal_dim', type=int, default=16, help='temporal feature dimension'
  )
  parser.add_argument(
      '--hidden_dim', type=int, default=64, help='hidden feature dimension'
  )

  # optimization
  parser.add_argument(
      '--num_workers', type=int, default=10, help='data loader num workers'
  )
  parser.add_argument(
      '--train_epochs', type=int, default=100, help='train epochs'
  )
  parser.add_argument(
      '--batch_size', type=int, default=32, help='batch size of input data'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.0001,
      help='optimizer learning rate',
  )
  parser.add_argument(
      '--patience', type=int, default=5, help='number of epochs to early stop'
  )

  # save results
  parser.add_argument(
      '--result_path', default='result.csv', help='path to save result'
  )

  args = parser.parse_args()

  tf.keras.utils.set_random_seed(args.seed)

  return args


def main():
  args = parse_args()
  input_shape = None
  if "context" in args.exp:
    args.feature_type = "MS"
  else:
    args.feature_type = "M"

  if 'tsmixer' in args.model:
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}_nt{args.norm_type}_{args.activation}_nb{args.n_block}_dp{args.dropout}_fd{args.ff_dim}'
  elif args.model == 'full_linear':
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}'
  elif args.model == 'cnn':
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}_ks{args.kernel_size}'
#   elif args.model == 'TSMixer':
#     exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}_ks{args.kernel_size}'
  else:
    raise ValueError(f'Unknown model type: {args.model}')

  # load datasets
  data_loader = TSFDataLoader(
      args.data,
      args.batch_size,
      args.seq_len,
      args.pred_len,
      args.feature_type,
      args.target,
  )
  train_data = data_loader.get_train()
  val_data = data_loader.get_val()
  test_data = data_loader.get_test()
#   pdb.set_trace()

  if not os.path.exists("./experiments/"+args.exp):
    os.makedirs("./experiments/"+args.exp)  
  # load context embeddings

  # train model
  if "context" in args.exp:
    input_shape = (args.seq_len, data_loader.n_feature+384)
  else:
    input_shape = (args.seq_len, data_loader.n_feature)
  if 'tsmixer' in args.model:
    build_model = getattr(models, args.model).build_model
    model = build_model(
        input_shape=input_shape,
        pred_len=args.pred_len,
        norm_type=args.norm_type,
        activation=args.activation,
        dropout=args.dropout,
        n_block=args.n_block,
        ff_dim=args.ff_dim,
        target_slice=data_loader.target_slice,
    )
  elif args.model == 'full_linear':
    model = models.full_linear.Model(
        n_channel=data_loader.n_feature,
        pred_len=args.pred_len,
    )
  elif args.model == 'cnn':
    model = models.cnn.Model(
        n_channel=data_loader.n_feature,
        pred_len=args.pred_len,
        kernel_size=args.kernel_size,
    )
  else:
    raise ValueError(f'Model not supported: {args.model}')

  optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
  model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
  checkpoint_path = os.path.join(args.checkpoint_dir, f'{exp_id}_best')
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      verbose=1,
      save_best_only=True,
      save_weights_only=True,
  )
  early_stop_callback = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss', patience=args.patience
  )
  start_training_time = time.time()
  history = model.fit(
      train_data,
      epochs=args.train_epochs,
      validation_data=val_data,
      callbacks=[checkpoint_callback, early_stop_callback],
  )
#   def train_step(model, input, loss_function, optimizer):
#     # GradientTape for automatic differentiation.
#     with tf.GradientTape() as tape:
#         for data in input:
#             pdb.set_trace()
#             prediction = model(data[0])
#             loss = loss_function(data[1], prediction)
#             gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss 
#   loss_ = train_step(model, train_data, tf.keras.losses.MeanSquaredError, optimizer)
  end_training_time = time.time()
  elasped_training_time = end_training_time - start_training_time
  print(f'Training finished in {elasped_training_time} seconds')

  # evaluate best model
  best_epoch = np.argmin(history.history['val_loss'])
  model.load_weights(checkpoint_path)
  test_result = model.evaluate(test_data)
  test_data_list = []
  test_data_history_list = []

  for element in test_data: test_data_list.extend(element[1])
  for element in test_data: test_data_history_list.extend(element[0])

  test_data_arr = np.array(test_data_list)
  test_data_hist_arr = np.array(test_data_history_list)
  
  predictions = model.predict(test_data)
  for i in range(5):
    test_data_hist_arr_inv_trans = data_loader.inverse_transform(test_data_hist_arr[i,:,0].reshape(-1,1)).astype(int)
    test_data_arr_inv_trans = data_loader.inverse_transform(test_data_arr[i,:,:]).astype(int)
    predictions_inv_trans = data_loader.inverse_transform(predictions[i,:,:]).astype(int)
    fig, ax = plt.subplots()
    ax.plot(np.arange(0,60), np.concatenate((test_data_hist_arr_inv_trans ,test_data_arr_inv_trans)), c="tab:green")
    ax.plot(np.arange(50,60), predictions_inv_trans, c ="tab:orange")
    plt.savefig("./experiments/"+args.exp+"/plot_"+str(i)+".png")
    # pdb.set_trace()


  if args.delete_checkpoint:
    for f in glob.glob(checkpoint_path + '*'):
      os.remove(f)

  # save result to csv
  data = {
      'data': [args.data],
      'exp' : [args.exp],
      'model': [args.model],
      'seq_len': [args.seq_len],
      'pred_len': [args.pred_len],
      'lr': [args.learning_rate],
      'mse': [test_result[0]],
      'mae': [test_result[1]],
      'val_mse': [history.history['val_loss'][best_epoch]],
      'val_mae': [history.history['val_mae'][best_epoch]],
      'train_mse': [history.history['loss'][best_epoch]],
      'train_mae': [history.history['mae'][best_epoch]],
      'training_time': elasped_training_time,
      'norm_type': args.norm_type,
      'activation': args.activation,
      'n_block': args.n_block,
      'dropout': args.dropout,
  }
  if 'TSMixer' in args.model:
    data['ff_dim'] = args.ff_dim

  df = pd.DataFrame(data)
  if os.path.exists(args.result_path):
    df.to_csv(args.result_path, mode='a', index=False, header=False)
  else:
    df.to_csv(args.result_path, mode='w', index=False, header=True)


if __name__ == '__main__':
  main()
