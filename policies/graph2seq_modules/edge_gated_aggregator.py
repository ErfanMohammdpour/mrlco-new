import tensorflow as tf
from .layers import Layer
from .inits import glorot, zeros

class EdgeGatedAggregator(Layer):
    """
    Edge-Gated Graph Convolution Aggregator.
    Each edge has its own sigmoid/attention gate.
    """
    
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0, bias=True, act=tf.nn.relu,
                 name=None, concat=False, edge_type_dim=None, **kwargs):
        super(EdgeGatedAggregator, self).__init__(**kwargs)
        
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.edge_type_dim = edge_type_dim  # For different edge types (normal vs virtual node)
        
        if name is not None:
            name = '/' + name
        else:
            name = ''
            
        if neigh_input_dim == None:
            neigh_input_dim = input_dim
            
        if concat:
            self.output_dim = 2 * output_dim
        else:
            self.output_dim = output_dim
            
        with tf.variable_scope(self.name + name + '_vars'):
            # Standard GCN weights
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                               name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                              name='self_weights')
            
            # Edge gating weights
            # Gate takes concatenated [source_features, target_features] as input
            gate_input_dim = neigh_input_dim + input_dim
            self.vars['edge_gate_weights'] = glorot([gate_input_dim, 1],
                                                    name='edge_gate_weights')
            self.vars['edge_gate_bias'] = zeros([1], name='edge_gate_bias')
            
            # Optional: separate gate weights for different edge types
            if edge_type_dim is not None:
                self.vars['edge_type_weights'] = glorot([edge_type_dim, 1],
                                                        name='edge_type_weights')
            
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')
                
        self.input_dim = input_dim
        self.neigh_input_dim = neigh_input_dim
        
    def _call(self, inputs):
        """
        inputs: (self_vecs, neigh_vecs, edge_types[optional])
        self_vecs: [batch_size * num_nodes, input_dim]
        neigh_vecs: [batch_size * num_nodes, num_neighbors, neigh_input_dim]
        edge_types: [batch_size * num_nodes, num_neighbors, edge_type_dim] (optional)
        """
        if len(inputs) == 3:
            self_vecs, neigh_vecs, edge_types = inputs
        else:
            self_vecs, neigh_vecs = inputs
            edge_types = None
            
        # Apply dropout
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        
        # Get dimensions
        num_nodes = tf.shape(self_vecs)[0]
        num_neighbors = tf.shape(neigh_vecs)[1]
        
        # Expand self_vecs to match neighbor dimension
        # [num_nodes, 1, input_dim] -> [num_nodes, num_neighbors, input_dim]
        self_vecs_expanded = tf.tile(tf.expand_dims(self_vecs, 1), [1, num_neighbors, 1])
        
        # Concatenate source and target features for edge gating
        # [num_nodes, num_neighbors, neigh_input_dim + input_dim]
        edge_features = tf.concat([neigh_vecs, self_vecs_expanded], axis=-1)
        
        # Compute edge gates
        # [num_nodes, num_neighbors, 1]
        edge_gates = tf.matmul(edge_features, self.vars['edge_gate_weights'])
        edge_gates = edge_gates + self.vars['edge_gate_bias']
        
        # Add edge type contribution if provided
        if edge_types is not None and self.edge_type_dim is not None:
            edge_type_contribution = tf.matmul(edge_types, self.vars['edge_type_weights'])
            edge_gates = edge_gates + edge_type_contribution
            
        # Apply sigmoid to get gate values in [0, 1]
        edge_gates = tf.nn.sigmoid(edge_gates)
        
        # Apply gates to neighbor features
        # [num_nodes, num_neighbors, neigh_input_dim]
        gated_neigh_vecs = neigh_vecs * edge_gates
        
        # Aggregate gated neighbor features (mean aggregation)
        # [num_nodes, neigh_input_dim]
        neigh_means = tf.reduce_mean(gated_neigh_vecs, axis=1)
        
        # Transform aggregated neighbors
        # [num_nodes, output_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])
        
        # Transform self features
        # [num_nodes, output_dim]
        from_self = tf.matmul(self_vecs, self.vars['self_weights'])
        
        # Combine
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)
            
        # Add bias
        if self.bias:
            output += self.vars['bias']
            
        # Apply activation
        return self.act(output)