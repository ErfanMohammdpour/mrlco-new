# Import TF2 compatibility layer
from compat.mpi_adam_optimizer import MpiAdamOptimizer, create_mpi_adam_optimizer

# Re-export for backwards compatibility
__all__ = ['MpiAdamOptimizer', 'create_mpi_adam_optimizer']
