import tensorflow as tf
from abc import ABC, abstractmethod

class BaseEncoder(ABC):
    """Abstract base class for sequence encoders in MRLCO."""
    
    @abstractmethod
    def encode(self, encoder_inputs):
        """Build the encoder graph and return (outputs, state).
        
        Args:
            encoder_inputs: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            tuple: (encoder_outputs, encoder_state)
                - encoder_outputs: Tensor of shape [batch_size, seq_len, output_dim]
                - encoder_state: Compatible state for decoder initialization
        """
        pass

    @abstractmethod
    def get_output_dim(self):
        """Return the final output feature dimension of the encoder.
        
        Returns:
            int: Output dimension size
        """
        pass