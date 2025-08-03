"""Distributed training utilities for TF1-style code compatibility"""
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Any


class DistributedTF1Trainer:
    """
    Wrapper for distributed training with TF1-style code that uses sessions and placeholders.
    This allows us to leverage multiple GPUs while maintaining compatibility with legacy code.
    """
    
    def __init__(self, strategy: tf.distribute.Strategy, device_info: Dict[str, Any]):
        self.strategy = strategy
        self.device_info = device_info
        self.num_replicas = strategy.num_replicas_in_sync
        
    def distributed_train_step(self, sess, feed_dict: Dict, train_ops: List, 
                              fetch_ops: List = None) -> Tuple[List, List]:
        """
        Execute training operations across distributed devices.
        
        For TF1 compatibility, we manually place operations on different devices
        rather than using strategy.run() which requires TF2 functions.
        
        Args:
            sess: TF1 session
            feed_dict: Dictionary of placeholder values
            train_ops: List of training operations to run
            fetch_ops: Optional list of tensors to fetch values from
            
        Returns:
            Tuple of (train results, fetched values)
        """
        if self.num_replicas == 1:
            # Single device - run normally
            if fetch_ops:
                results = sess.run(train_ops + fetch_ops, feed_dict=feed_dict)
                train_results = results[:len(train_ops)]
                fetch_results = results[len(train_ops):]
                return train_results, fetch_results
            else:
                train_results = sess.run(train_ops, feed_dict=feed_dict)
                return train_results, []
        
        # Multi-GPU: Split data and run on each device
        # For TF1, we need to manually handle data distribution
        batch_size = self._get_batch_size_from_feed_dict(feed_dict)
        if batch_size is None or batch_size < self.num_replicas:
            # Batch too small for distribution, run on single GPU
            with tf.device(self.device_info['selected_devices'][0]):
                if fetch_ops:
                    results = sess.run(train_ops + fetch_ops, feed_dict=feed_dict)
                    train_results = results[:len(train_ops)]
                    fetch_results = results[len(train_ops):]
                    return train_results, fetch_results
                else:
                    train_results = sess.run(train_ops, feed_dict=feed_dict)
                    return train_results, []
        
        # Split batch across devices
        per_replica_batch_size = batch_size // self.num_replicas
        
        # Create per-device feed dicts
        device_feed_dicts = []
        for i in range(self.num_replicas):
            start_idx = i * per_replica_batch_size
            end_idx = (i + 1) * per_replica_batch_size if i < self.num_replicas - 1 else batch_size
            
            device_feed_dict = {}
            for key, value in feed_dict.items():
                if isinstance(value, np.ndarray) and len(value.shape) > 0:
                    # Slice along batch dimension
                    device_feed_dict[key] = value[start_idx:end_idx]
                else:
                    # Scalar or non-batchable value
                    device_feed_dict[key] = value
            device_feed_dicts.append(device_feed_dict)
        
        # Run operations on each device and aggregate results
        all_train_results = []
        all_fetch_results = []
        
        for i, device in enumerate(self.device_info['selected_devices'][:self.num_replicas]):
            with tf.device(device):
                if fetch_ops:
                    results = sess.run(train_ops + fetch_ops, feed_dict=device_feed_dicts[i])
                    all_train_results.append(results[:len(train_ops)])
                    all_fetch_results.append(results[len(train_ops):])
                else:
                    train_results = sess.run(train_ops, feed_dict=device_feed_dicts[i])
                    all_train_results.append(train_results)
        
        # Aggregate results (average for losses, concatenate for predictions)
        aggregated_train = []
        for i in range(len(train_ops)):
            values = [result[i] for result in all_train_results]
            if all(np.isscalar(v) or (isinstance(v, np.ndarray) and v.size == 1) for v in values):
                # Scalar values - average them
                aggregated_train.append(np.mean(values))
            else:
                # Array values - concatenate along batch dimension
                aggregated_train.append(np.concatenate(values, axis=0))
        
        aggregated_fetch = []
        if fetch_ops:
            for i in range(len(fetch_ops)):
                values = [result[i] for result in all_fetch_results]
                if all(np.isscalar(v) or (isinstance(v, np.ndarray) and v.size == 1) for v in values):
                    aggregated_fetch.append(np.mean(values))
                else:
                    aggregated_fetch.append(np.concatenate(values, axis=0))
        
        return aggregated_train, aggregated_fetch
    
    def _get_batch_size_from_feed_dict(self, feed_dict: Dict) -> int:
        """Extract batch size from feed_dict by looking at tensor shapes"""
        for value in feed_dict.values():
            if isinstance(value, np.ndarray) and len(value.shape) > 0:
                return value.shape[0]
        return None
    
    def create_distributed_dataset(self, data: np.ndarray, batch_size: int, 
                                  shuffle: bool = True, seed: int = None) -> tf.data.Dataset:
        """
        Create a distributed dataset optimized for multi-GPU training.
        
        Args:
            data: Numpy array of data
            batch_size: Global batch size (will be divided across replicas)
            shuffle: Whether to shuffle the data
            seed: Random seed for shuffling
            
        Returns:
            tf.data.Dataset configured for distributed training
        """
        dataset = tf.data.Dataset.from_tensor_slices(data)
        
        if shuffle:
            buffer_size = min(10000, len(data))
            dataset = dataset.shuffle(buffer_size, seed=seed)
        
        # Use drop_remainder to ensure even distribution across replicas
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        # Optimize pipeline
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Distribute dataset across replicas
        if hasattr(self.strategy, 'experimental_distribute_dataset'):
            dataset = self.strategy.experimental_distribute_dataset(dataset)
        
        return dataset


def create_mirrored_variables(var_list: List[tf.Variable], devices: List[str]) -> Dict[str, List[tf.Variable]]:
    """
    Create mirrored copies of variables across devices for manual distribution.
    
    Args:
        var_list: List of TF1 variables to mirror
        devices: List of device strings
        
    Returns:
        Dictionary mapping variable names to list of device copies
    """
    mirrored_vars = {}
    
    for var in var_list:
        var_name = var.name
        mirrored_vars[var_name] = []
        
        # First device uses the original variable
        mirrored_vars[var_name].append(var)
        
        # Create copies for other devices
        for i in range(1, len(devices)):
            with tf.device(devices[i]):
                # Create a copy with a unique name
                copy_name = f"{var.name.split(':')[0]}_device{i}:0"
                var_copy = tf.Variable(
                    initial_value=var.initialized_value(),
                    trainable=var.trainable,
                    name=copy_name
                )
                mirrored_vars[var_name].append(var_copy)
    
    return mirrored_vars


def aggregate_gradients(gradient_list: List[List[Tuple[tf.Tensor, tf.Variable]]]) -> List[Tuple[tf.Tensor, tf.Variable]]:
    """
    Aggregate gradients from multiple devices.
    
    Args:
        gradient_list: List of (gradient, variable) tuples from each device
        
    Returns:
        List of averaged (gradient, variable) tuples
    """
    num_devices = len(gradient_list)
    if num_devices == 1:
        return gradient_list[0]
    
    # Group gradients by variable
    aggregated = []
    num_grads = len(gradient_list[0])
    
    for i in range(num_grads):
        grads = [grad_list[i][0] for grad_list in gradient_list if grad_list[i][0] is not None]
        var = gradient_list[0][i][1]
        
        if not grads:
            aggregated.append((None, var))
        else:
            # Average gradients
            avg_grad = tf.add_n(grads) / float(len(grads))
            aggregated.append((avg_grad, var))
    
    return aggregated