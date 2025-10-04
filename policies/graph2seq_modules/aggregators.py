import tensorflow as tf
from .layers import Layer, Dense
from .inits import glorot, zeros
from .pooling import mean_pool


class AttentiveStatisticsAggregator(Layer):
    """
    Attentive mean+std pooling over neighbors, then combine with self vector.
    Inputs:
      - (self_vecs, neigh_vecs) or (self_vecs, neigh_vecs, neigh_mask)
        self_vecs:  [B, D_self]
        neigh_vecs: [B, K, D_neigh]
        neigh_mask: [B, K] with 1 for real, 0 for padding (optional)
    Output:
      - [B, out] if concat=False
      - [B, 2*out] if concat=True
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0.0, attn_dropout=0.0, bias=True, act=tf.nn.relu,
                 name=None, concat=False, mode="train", attn_temp=1.0,
                 use_small_sample_correction=False, use_gate=False, **kwargs):
        super(AttentiveStatisticsAggregator, self).__init__(**kwargs)

        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)
        self.bias = bias
        self.act = act
        self.concat = concat
        self.mode = mode
        self.attn_temp = float(max(attn_temp, 1e-6))
        self.use_small_sample_correction = use_small_sample_correction
        self.use_gate = use_gate

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        # Define output dimensions before creating variables
        self.single_branch_out = output_dim
        self.final_out = 2 * output_dim if concat else output_dim

        with tf.variable_scope(self.name + name + '_vars'):
            # scalar attention logit per neighbor: neigh_vecs @ attn_w + attn_b
            self.vars['attn_w'] = glorot([neigh_input_dim, 1], name='attn_w')
            self.vars['attn_b'] = zeros([1], name='attn_b')

            # project neighbor stats [mu, std] (2*D_neigh) -> out
            self.vars['neigh_stats_w'] = glorot([2 * neigh_input_dim, output_dim], name='neigh_stats_w')
            self.vars['neigh_stats_b'] = zeros([output_dim], name='neigh_stats_b')

            # project self vector -> out
            self.vars['self_w'] = glorot([input_dim, output_dim], name='self_w')
            self.vars['self_b'] = zeros([output_dim], name='self_b')

            if self.bias:
                self.vars['bias'] = zeros([self.final_out], name='bias')

        self.input_dim = input_dim
        self.output_dim = self.final_out
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        # Unpack
        if len(inputs) == 3:
            self_vecs, neigh_vecs, neigh_mask = inputs
            use_mask = True
        else:
            self_vecs, neigh_vecs = inputs
            neigh_mask = None
            use_mask = False

        # Input dropout (train only)
        if self.mode == "train" and self.dropout > 0.0:
            # feature-wise dropout over neighbors (same mask across features per neighbor slot)
            neigh_vecs = tf.nn.dropout(
                neigh_vecs,
                keep_prob=1.0 - self.dropout,
                noise_shape=tf.stack([tf.shape(neigh_vecs)[0], tf.shape(neigh_vecs)[1], 1])
            )
            self_vecs = tf.nn.dropout(self_vecs, keep_prob=1.0 - self.dropout)

        # Attention logits: [B, K, 1]
        attn_logits = tf.tensordot(neigh_vecs, self.vars['attn_w'], axes=1) + self.vars['attn_b']  # [B,K,1]

        # Mask padded neighbors
        if use_mask:
            neg_inf = tf.constant(-1e9, dtype=attn_logits.dtype)
            attn_logits = attn_logits + tf.expand_dims(
                (1.0 - tf.cast(neigh_mask, attn_logits.dtype)) * neg_inf, -1
            )

        # Weights with temperature
        attn_weights = tf.nn.softmax(attn_logits / self.attn_temp, axis=1)  # [B,K,1]
        
        if self.mode == "train" and self.attn_dropout > 0.0:
            attn_weights = tf.nn.dropout(attn_weights, keep_prob=1.0 - self.attn_dropout)

        # Weighted mean and std
        mu = tf.reduce_sum(attn_weights * neigh_vecs, axis=1)  # [B, D_neigh]
        mu_exp = tf.expand_dims(mu, axis=1)                    # [B,1,D_neigh]
        var_num = tf.reduce_sum(attn_weights * tf.square(neigh_vecs - mu_exp), axis=1)  # [B, D_neigh]
        
        # Small-sample variance correction
        if self.use_small_sample_correction:
            w = tf.squeeze(attn_weights, -1)  # [B,K]
            w2_sum = tf.reduce_sum(tf.square(w), axis=1, keepdims=True)  # [B,1]
            var_corr = var_num / (1.0 - w2_sum + 1e-6)
            var = var_corr
        else:
            var = var_num
            
        std = tf.sqrt(var + 1e-6)

        # Neighbor projection
        neigh_stats = tf.concat([mu, std], axis=1)  # [B, 2*D_neigh]
        from_neighs = tf.matmul(neigh_stats, self.vars['neigh_stats_w']) + self.vars['neigh_stats_b']  # [B,out]

        # Self projection
        from_self = tf.matmul(self_vecs, self.vars['self_w']) + self.vars['self_b']  # [B,out]

        # Combine
        if not self.concat:
            if self.use_gate:
                # Optional gate for self vs neighbors
                gate_w = self.vars.get('gate_w') or self.vars.setdefault('gate_w', glorot([self.input_dim, 1], name='gate_w'))
                gate_b = self.vars.get('gate_b') or self.vars.setdefault('gate_b', zeros([1], name='gate_b'))
                gate = tf.sigmoid(tf.matmul(self_vecs, gate_w) + gate_b)  # [B,1]
                output = gate * from_self + (1.0 - gate) * from_neighs
            else:
                output = tf.add_n([from_self, from_neighs])  # [B,out]
        else:
            output = tf.concat([from_self, from_neighs], axis=1)  # [B,2*out]

        if self.bias:
            output += self.vars['bias']

        # Post-aggregation dropout (often helpful)
        if self.mode == "train" and self.dropout > 0.0:
            output = tf.nn.dropout(output, keep_prob=1.0 - self.dropout)

        return self.act(output)

        
class GatedMeanAggregator(Layer):
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0, bias=True, act=tf.nn.relu,
            name=None, concat=False, **kwargs):
        super(GatedMeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        if concat:
            self.output_dim = 2 * output_dim

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

            self.vars['gate_weights'] = glorot([2*output_dim, 2*output_dim],
                                                name='gate_weights')
            self.vars['gate_bias'] = zeros([2*output_dim], name='bias')


        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)

        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        gate = tf.concat([from_self, from_neighs], axis=1)
        gate = tf.matmul(gate, self.vars["gate_weights"]) + self.vars["gate_bias"]
        gate = tf.nn.relu(gate)

        return gate*self.act(output)

class MeanAggregator(Layer):
    """Aggregates via mean followed by matmul and non-linearity."""

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0, bias=True, act=tf.nn.relu,
            name=None, concat=False, mode="train", **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.mode = mode

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        if concat:
            self.output_dim = 2 * output_dim

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs, neigh_len = inputs

        if self.mode == "train":
            neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
            self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)

        # reduce_mean performs better than mean_pool
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
        # neigh_means = mean_pool(neigh_vecs, neigh_len)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions."""
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=True, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        if concat:
            self.output_dim = 2 * output_dim

        if model_size == "small":
            hidden_dim = self.hidden_dim = 50
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 50

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim, output_dim=hidden_dim, act=tf.nn.relu,
                                     dropout=dropout, sparse_inputs=False, logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):

            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim], name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim], name='self_weights')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]

        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
        return self.act(output)