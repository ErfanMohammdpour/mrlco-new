import numpy as np
import tensorflow as tf
from compat import rnn as contrib_rnn
import utils.logger as logger

tf.get_logger().setLevel('WARNING')

def _single_cell(unit_type, num_units, forget_bias, dropout, mode,
                 residual_connection=False, device_str=None, residual_fn=None):
  """Create an instance of a single RNN cell."""
  # dropout (= 1 - keep_prob) is set to 0 during eval and infer
  # MIGRATION: Using string mode instead of ModeKeys enum
  dropout = dropout if mode == 'train' else 0.0

  # Cell Type
  if unit_type == "lstm":
    single_cell = contrib_rnn.BasicLSTMCell(
        num_units,
        forget_bias=forget_bias)
  elif unit_type == "gru":
    single_cell = contrib_rnn.GRUCell(num_units)
  elif unit_type == "layer_norm_lstm":
    single_cell = contrib_rnn.LayerNormBasicLSTMCell(
        num_units,
        forget_bias=forget_bias,
        layer_norm=True)
  elif unit_type == "nas":
    single_cell = contrib_rnn.NASCell(num_units)
  else:
    raise ValueError("Unknown unit type %s!" % unit_type)

  if dropout > 0.0:
    single_cell = contrib_rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))

  # Residual
  if residual_connection:
    single_cell = contrib_rnn.ResidualWrapper(
        single_cell, residual_fn=residual_fn)

  return single_cell


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, num_gpus, base_gpu=0,
               single_cell_fn=None, residual_fn=None):
  if not single_cell_fn:
    single_cell_fn = _single_cell

  cell_list = []
  for i in range(num_layers):
    single_cell = single_cell_fn(
        unit_type=unit_type,
        num_units=num_units,
        forget_bias=forget_bias,
        dropout=dropout,
        mode=mode,
        residual_connection=(i >= num_layers - num_residual_layers),
        residual_fn=residual_fn
    )
    cell_list.append(single_cell)

  return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, num_gpus, base_gpu=0,
                    single_cell_fn=None):

  cell_list = _cell_list(unit_type=unit_type,
                         num_units=num_units,
                         num_layers=num_layers,
                         num_residual_layers=num_residual_layers,
                         forget_bias=forget_bias,
                         dropout=dropout,
                         mode=mode,
                         num_gpus=num_gpus,
                         base_gpu=base_gpu,
                         single_cell_fn=single_cell_fn)

  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  else:  # Multi layers
    return contrib_rnn.MultiRNNCell(cell_list)
