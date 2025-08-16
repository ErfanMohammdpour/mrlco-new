import tensorflow as tf
from .base_encoder import BaseEncoder
from . import model_helper


class RNNEncoder(BaseEncoder):
    """RNN encoder that implements BaseEncoder interface."""
    
    def __init__(self, unit_type="lstm", hidden_dim=256, num_layers=2, bidirectional=False, dropout=0.0, mode='train'):
        self.unit_type = unit_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.mode = mode
        
    def encode(self, encoder_inputs):
        """Build RNN encoder graph and return (outputs, state)."""
        if self.bidirectional:
            return self._create_bidirectional_encoder(encoder_inputs)
        else:
            return self._create_encoder(encoder_inputs)
            
    def get_output_dim(self):
        """Return the final output feature dimension of the encoder."""
        return self.hidden_dim * (2 if self.bidirectional else 1)
        
    def _create_encoder(self, encoder_inputs):
        """Create unidirectional RNN encoder."""
        encoder_cell = model_helper.create_rnn_cell(
            unit_type=self.unit_type,
            num_units=self.hidden_dim,
            num_layers=self.num_layers,
            num_residual_layers=0,
            forget_bias=1.0,
            dropout=self.dropout,
            mode=self.mode,
            num_gpus=1,
            base_gpu=0
        )
        
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell=encoder_cell,
            inputs=encoder_inputs,
            dtype=tf.float32,
            time_major=False,
            swap_memory=True
        )
        
        return encoder_outputs, encoder_state
        
    def _create_bidirectional_encoder(self, encoder_inputs):
        """Create bidirectional RNN encoder."""
        num_bi_layers = max(1, int(self.num_layers / 2))
        
        forward_cell = model_helper.create_rnn_cell(
            unit_type=self.unit_type,
            num_units=self.hidden_dim,
            num_layers=num_bi_layers,
            num_residual_layers=0,
            forget_bias=1.0,
            dropout=self.dropout,
            mode=self.mode,
            num_gpus=1,
            base_gpu=0
        )
        
        backward_cell = model_helper.create_rnn_cell(
            unit_type=self.unit_type,
            num_units=self.hidden_dim,
            num_layers=num_bi_layers,
            num_residual_layers=0,
            forget_bias=1.0,
            dropout=self.dropout,
            mode=self.mode,
            num_gpus=1,
            base_gpu=0
        )
        
        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            forward_cell,
            backward_cell,
            inputs=encoder_inputs,
            time_major=False,
            swap_memory=True,
            dtype=tf.float32
        )
        
        encoder_outputs = tf.concat(bi_outputs, -1)
        
        if num_bi_layers == 1:
            if self.unit_type == "lstm":
                fw_state = bi_state[0]
                bw_state = bi_state[1]
                encoder_state = tf.nn.rnn_cell.LSTMStateTuple(
                    c=tf.concat([fw_state.c, bw_state.c], -1),
                    h=tf.concat([fw_state.h, bw_state.h], -1)
                )
            else:
                encoder_state = tf.concat(bi_state, -1)
        else:
            encoder_state = []
            for layer_id in range(num_bi_layers):
                if self.unit_type == "lstm":
                    fw_state = bi_state[0][layer_id]
                    bw_state = bi_state[1][layer_id]
                    layer_state = tf.nn.rnn_cell.LSTMStateTuple(
                        c=tf.concat([fw_state.c, bw_state.c], -1),
                        h=tf.concat([fw_state.h, bw_state.h], -1)
                    )
                    encoder_state.append(layer_state)
                else:
                    encoder_state.append(bi_state[0][layer_id])
                    encoder_state.append(bi_state[1][layer_id])
            encoder_state = tuple(encoder_state)
            
        return encoder_outputs, encoder_state