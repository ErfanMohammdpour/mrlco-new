import numpy as np
import tensorflow as tf

# Stub for single-process training - no MPI dependency
class MpiAdamOptimizer(tf.keras.optimizers.Adam):
    """TF2.19 Adam optimizer - MPI functionality removed for single-process training."""
    def __init__(self, comm=None, **kwargs):
        # Ignore comm parameter for single-process training
        super().__init__(**kwargs)
        if comm is not None:
            print("Warning: MPI comm parameter ignored in single-process mode")
    
    # Just use the standard Adam optimizer functionality
    # No need to override apply_gradients for single process

def create_mpi_adam_optimizer(comm=None, learning_rate=1e-3, **kwargs):
    """Factory function to create Adam optimizer (MPI functionality removed)"""
    if comm is not None:
        print("Warning: MPI not available, using regular Adam optimizer")
    return MpiAdamOptimizer(comm=None, learning_rate=learning_rate, **kwargs)

# Re-export for backwards compatibility
__all__ = ['MpiAdamOptimizer', 'create_mpi_adam_optimizer']