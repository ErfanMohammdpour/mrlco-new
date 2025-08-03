import tensorflow as tf
import numpy as np
from .layers import Layer, Dense
from .inits import glorot, zeros
from .pooling import mean_pool

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

        with tf.name_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = tf.Variable(
                tf.keras.initializers.GlorotUniform()(shape=(neigh_input_dim, output_dim)) * 0.5,
                dtype=tf.float32,
                name='neigh_weights'
            )
            self.vars['self_weights'] = tf.Variable(
                tf.keras.initializers.GlorotUniform()(shape=(input_dim, output_dim)) * 0.5,
                dtype=tf.float32,
                name='self_weights'
            )
            if self.bias:
                self.vars['bias'] = tf.Variable(
                    tf.zeros([self.output_dim], dtype=tf.float32), 
                    name='bias'
                )

            self.vars['gate_weights'] = tf.Variable(
                tf.keras.initializers.GlorotUniform()(shape=(2*output_dim, 2*output_dim)) * 0.5,
                dtype=tf.float32,
                name='gate_weights'
            )
            self.vars['gate_bias'] = tf.Variable(
                tf.zeros([2*output_dim], dtype=tf.float32), 
                name='gate_bias'
            )


        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, rate=self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, rate=self.dropout)

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

class MeanAggregator(tf.keras.layers.Layer):
    """Aggregates via mean followed by matmul and non-linearity."""

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0, bias=True, act=tf.nn.relu,
            name=None, concat=False, mode="train", **kwargs):
        super(MeanAggregator, self).__init__(name=name, **kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.mode = mode

        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        if concat:
            self.output_dim = 2 * output_dim
        else:
            self.output_dim = output_dim

        self.input_dim = input_dim
        self.base_output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def build(self, input_shape):
        """Build the layer - create weights using Keras add_weight"""
        super(MeanAggregator, self).build(input_shape)
        
        # Handle different input_shape formats (tuple of shapes vs single shape)
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
            # input_shape is a tuple/list of shapes: (self_shape, neigh_shape, len_shape)
            pass  # Use class-level dimensions
        else:
            # Single input shape - use it directly
            pass
        
        # Use Keras add_weight to properly handle variable creation
        self.neigh_weights = self.add_weight(
            name='neigh_weights',
            shape=(self.neigh_input_dim, self.base_output_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            dtype=tf.float32
        )
        
        self.self_weights = self.add_weight(
            name='self_weights',
            shape=(self.input_dim, self.base_output_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            dtype=tf.float32
        )
        
        if self.bias:
            self.bias_weights = self.add_weight(
                name='bias',
                shape=(self.output_dim,),
                initializer='zeros',
                trainable=True,
                dtype=tf.float32
            )

    def call(self, inputs, training=None):
        self_vecs, neigh_vecs, neigh_len = inputs

        if training and self.mode == "train":
            neigh_vecs = tf.nn.dropout(neigh_vecs, rate=self.dropout)
            self_vecs = tf.nn.dropout(self_vecs, rate=self.dropout)

        # reduce_mean performs better than mean_pool
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
        # neigh_means = mean_pool(neigh_vecs, neigh_len)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.neigh_weights)

        from_self = tf.matmul(self_vecs, self.self_weights)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.bias_weights

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

        with tf.name_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = tf.Variable(
                tf.keras.initializers.GlorotUniform()(shape=(hidden_dim, output_dim)) * 0.5,
                dtype=tf.float32,
                name='neigh_weights'
            )

            self.vars['self_weights'] = tf.Variable(
                tf.keras.initializers.GlorotUniform()(shape=(input_dim, output_dim)) * 0.5,
                dtype=tf.float32,
                name='self_weights'
            )

            if self.bias:
                self.vars['bias'] = tf.Variable(
                    tf.zeros([self.output_dim], dtype=tf.float32), 
                    name='bias'
                )

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