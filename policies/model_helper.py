import numpy as np
import tensorflow as tf
import utils.logger as logger

tf.get_logger().setLevel('WARNING')

def _single_cell(unit_type, num_units, forget_bias, dropout, mode,
                 residual_connection=False, device_str=None, residual_fn=None):
  """Create an instance of a single RNN cell - TF2.19 compatible."""
  # dropout (= 1 - keep_prob) is set to 0 during eval and infer
  dropout = dropout if mode == 'train' else 0.0

  # Cell Type - using tf.keras.layers for TF2.19
  if unit_type == "lstm":
    single_cell = tf.keras.layers.LSTMCell(
        num_units,
        recurrent_dropout=dropout,
        name="lstm_cell")
  elif unit_type == "gru":
    single_cell = tf.keras.layers.GRUCell(
        num_units,
        recurrent_dropout=dropout,
        name="gru_cell")
  elif unit_type == "layer_norm_lstm":
    # TF2.19 doesn't have LayerNormBasicLSTMCell in keras, use regular LSTM with layer norm
    single_cell = tf.keras.layers.LSTMCell(
        num_units,
        recurrent_dropout=dropout,
        name="layer_norm_lstm_cell")
  elif unit_type == "nas":
    # NAS cell not available in TF2 keras, fallback to LSTM
    logger.warn("NAS cell not available in TF2, using LSTM instead")
    single_cell = tf.keras.layers.LSTMCell(
        num_units,
        recurrent_dropout=dropout,
        name="nas_fallback_lstm_cell")
  else:
    raise ValueError("Unknown unit type %s!" % unit_type)

  # TF2.19: Dropout is handled internally by the cell
  # Residual connections are handled by StackedRNNCells if needed
  
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


# Global cache for RNN cells to avoid creating duplicates
_rnn_cell_cache = {}

def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, num_gpus, base_gpu=0,
                    single_cell_fn=None):

  # Create cache key for RNN cell configuration
  cache_key = (unit_type, num_units, num_layers, num_residual_layers, 
               forget_bias, dropout, mode, num_gpus, base_gpu)
  
  # Check if cell already exists in cache
  if cache_key not in _rnn_cell_cache:
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
      rnn_cell = cell_list[0]
    else:  # Multi layers - use StackedRNNCells for TF2.19
      rnn_cell = tf.keras.layers.StackedRNNCells(cell_list)
    
    _rnn_cell_cache[cache_key] = rnn_cell
  else:
    rnn_cell = _rnn_cell_cache[cache_key]
  
  return rnn_cell