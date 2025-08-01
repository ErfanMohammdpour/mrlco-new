from .layers import Layer
import tensorflow as tf

class UniformNeighborSampler(Layer):
    """
       Uniformly samples neighbors.
       Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        adj_lists = tf.transpose(tf.transpose(adj_lists))
        
        # Get the actual width of adj_lists to avoid slice errors
        adj_width = tf.shape(adj_lists)[1]
        # Use minimum of requested samples and actual available columns
        actual_samples = tf.minimum(num_samples, adj_width)
        
        adj_lists = tf.slice(adj_lists, [0,0], [-1, actual_samples])
        return adj_lists