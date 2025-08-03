"""
TF2 compatibility layer for MpiAdamOptimizer
Migrates from tf.train.AdamOptimizer to tf.keras.optimizers.Adam
"""
import numpy as np
import tensorflow as tf
from mpi4py import MPI


class MpiAdamOptimizer(tf.keras.optimizers.Adam):
    """Adam optimizer that averages gradients across mpi processes.
    
    TF2 version that replaces tf.train.AdamOptimizer with tf.keras.optimizers.Adam
    """
    def __init__(self, comm, **kwargs):
        self.comm = comm
        super().__init__(**kwargs)
    
    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        """Override to average gradients across MPI processes before applying"""
        # Filter out None gradients
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        
        if not grads_and_vars:
            return super().apply_gradients([], name=name, **kwargs)
        
        # Average gradients across MPI processes
        averaged_grads_and_vars = self._average_gradients(grads_and_vars)
        
        # Apply the averaged gradients
        return super().apply_gradients(averaged_grads_and_vars, name=name, **kwargs)
    
    def _average_gradients(self, grads_and_vars):
        """Average gradients across MPI processes"""
        # Flatten all gradients
        flat_grads = []
        shapes = []
        for grad, var in grads_and_vars:
            flat_grad = tf.reshape(grad, [-1])
            flat_grads.append(flat_grad)
            shapes.append(tf.shape(grad))
        
        # Concatenate all gradients
        concat_grads = tf.concat(flat_grads, axis=0)
        
        # Convert to numpy for MPI operations
        def _mpi_average(grad_tensor):
            grad_np = grad_tensor.numpy()
            
            # MPI Allreduce to sum gradients across processes
            num_tasks = self.comm.Get_size()
            buf = np.zeros_like(grad_np)
            self.comm.Allreduce(grad_np, buf, op=MPI.SUM)
            
            # Average by dividing by number of processes
            buf = buf / float(num_tasks)
            
            return buf.astype(np.float32)
        
        # Use tf.py_function for MPI operations
        averaged_flat = tf.py_function(
            _mpi_average, 
            [concat_grads], 
            tf.float32
        )
        averaged_flat.set_shape(concat_grads.shape)
        
        # Split back into individual gradients
        sizes = [tf.reduce_prod(shape) for shape in shapes]
        split_grads = tf.split(averaged_flat, sizes, axis=0)
        
        # Reshape back to original shapes
        averaged_grads_and_vars = []
        for i, (_, var) in enumerate(grads_and_vars):
            reshaped_grad = tf.reshape(split_grads[i], shapes[i])
            averaged_grads_and_vars.append((reshaped_grad, var))
        
        return averaged_grads_and_vars


# Backwards compatibility function for TF1-style usage
def create_mpi_adam_optimizer(comm, learning_rate=0.001, **kwargs):
    """Factory function for creating MPI Adam optimizer with TF1-style interface"""
    return MpiAdamOptimizer(comm=comm, learning_rate=learning_rate, **kwargs)