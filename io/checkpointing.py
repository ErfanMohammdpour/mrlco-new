"""
Centralized checkpoint handling for TF2 migration
Provides unified interface for joblib and TF2 checkpoints
"""
import os
import joblib
import tensorflow as tf
from compat import checkpoint as compat_checkpoint


class CheckpointManager:
    """Manages checkpoints with support for both joblib and TF2 formats"""
    
    def __init__(self, model, checkpoint_dir="./checkpoints", legacy_format="joblib"):
        """
        Args:
            model: Keras model or object with variables
            checkpoint_dir: Directory to save checkpoints
            legacy_format: Format for backward compatibility ("joblib" or "tf2")
        """
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.legacy_format = legacy_format
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # TF2 checkpoint object
        self.tf_checkpoint = tf.train.Checkpoint(model=model)
        self.tf_checkpoint_manager = tf.train.CheckpointManager(
            self.tf_checkpoint,
            directory=os.path.join(checkpoint_dir, "tf2"),
            max_to_keep=5
        )
        
    def save(self, step=None, path=None):
        """Save checkpoint in specified format
        
        Args:
            step: Training step number
            path: Optional custom path (for joblib format)
        """
        if self.legacy_format == "joblib":
            if path is None:
                path = os.path.join(self.checkpoint_dir, f"model_{step}.pkl")
            self.save_joblib(path)
        else:
            self.save_tf2(step)
            
    def save_joblib(self, path):
        """Save in joblib format for backward compatibility"""
        variables = self.get_model_variables()
        compat_checkpoint.save_variables_joblib(variables, path)
        print(f"Saved joblib checkpoint to {path}")
        
    def save_tf2(self, step=None):
        """Save in TF2 format"""
        save_path = self.tf_checkpoint_manager.save(checkpoint_number=step)
        print(f"Saved TF2 checkpoint to {save_path}")
        return save_path
        
    def load(self, path=None, expect_partial=False):
        """Load checkpoint from specified path
        
        Args:
            path: Checkpoint path (auto-detects format)
            expect_partial: If True, suppress warnings about missing variables
        """
        if path is None:
            # Try to load latest TF2 checkpoint
            path = self.tf_checkpoint_manager.latest_checkpoint
            if path:
                return self.load_tf2(path, expect_partial)
            else:
                raise ValueError("No checkpoint path provided and no TF2 checkpoints found")
                
        # Detect format by extension
        if path.endswith('.pkl') or path.endswith('.ckpt'):
            return self.load_joblib(path)
        else:
            return self.load_tf2(path, expect_partial)
            
    def load_joblib(self, path):
        """Load from joblib format"""
        variables = self.get_model_variables()
        compat_checkpoint.load_variables_joblib(variables, path)
        print(f"Loaded joblib checkpoint from {path}")
        
    def load_tf2(self, path, expect_partial=False):
        """Load from TF2 format"""
        status = self.tf_checkpoint.restore(path)
        if expect_partial:
            status.expect_partial()
        else:
            # This will raise an error if variables are missing
            status.assert_consumed()
        print(f"Loaded TF2 checkpoint from {path}")
        return status
        
    def get_model_variables(self):
        """Get all variables from the model"""
        if hasattr(self.model, 'variables'):
            return self.model.variables
        elif hasattr(self.model, 'trainable_variables'):
            return self.model.trainable_variables
        else:
            raise ValueError("Model must have 'variables' or 'trainable_variables' attribute")
            
    def convert_joblib_to_tf2(self, joblib_path, tf2_path=None):
        """Convert a joblib checkpoint to TF2 format"""
        # Load joblib checkpoint
        self.load_joblib(joblib_path)
        
        # Save in TF2 format
        if tf2_path is None:
            tf2_path = self.save_tf2()
        else:
            self.tf_checkpoint.save(tf2_path)
            
        print(f"Converted {joblib_path} to TF2 format at {tf2_path}")
        return tf2_path
        
    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = {
            'joblib': [],
            'tf2': []
        }
        
        # List joblib checkpoints
        if os.path.exists(self.checkpoint_dir):
            for file in os.listdir(self.checkpoint_dir):
                if file.endswith('.pkl') or file.endswith('.ckpt'):
                    checkpoints['joblib'].append(os.path.join(self.checkpoint_dir, file))
                    
        # List TF2 checkpoints
        tf2_dir = os.path.join(self.checkpoint_dir, "tf2")
        if os.path.exists(tf2_dir):
            checkpoints['tf2'] = tf.train.get_checkpoint_state(tf2_dir).all_model_checkpoint_paths
            
        return checkpoints


# Convenience functions for backward compatibility
def save_checkpoint(model, path, legacy_format=True):
    """Save checkpoint with automatic format detection"""
    if legacy_format:
        variables = model.variables if hasattr(model, 'variables') else model.trainable_variables
        compat_checkpoint.save_variables_joblib(variables, path)
    else:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.save(path)
        
        
def load_checkpoint(model, path, expect_partial=False):
    """Load checkpoint with automatic format detection"""
    if path.endswith('.pkl') or path.endswith('.ckpt'):
        # Joblib format
        variables = model.variables if hasattr(model, 'variables') else model.trainable_variables
        compat_checkpoint.load_variables_joblib(variables, path)
    else:
        # TF2 format
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(path)
        if expect_partial:
            status.expect_partial()
        else:
            status.assert_consumed()
        return status