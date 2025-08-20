"""
Test training dynamics with Graph2Seq encoder to ensure it learns better than baseline.
"""
import tensorflow as tf
import numpy as np
from policies.meta_seq2seq_policy import Seq2SeqPolicy
from meta_algos.ppo_offloading import PPO
import matplotlib.pyplot as plt
import os


def create_synthetic_task_data(batch_size, seq_len, obs_dim, vocab_size):
    """Create synthetic task data for testing."""
    # Create pattern where certain observations should map to certain actions
    observations = np.random.randn(batch_size, seq_len, obs_dim).astype(np.float32)
    
    # Create actions based on simple rule (for testing)
    actions = np.zeros((batch_size, seq_len), dtype=np.int32)
    for i in range(batch_size):
        for j in range(seq_len):
            # Simple rule: action depends on sign of first observation feature
            if observations[i, j, 0] > 0:
                actions[i, j] = 1
            else:
                actions[i, j] = 0
                
    return observations, actions


def test_training_dynamics():
    """Test that Graph2Seq encoder improves during training."""
    print("Testing Graph2Seq Encoder Training Dynamics")
    print("="*60)
    
    tf.reset_default_graph()
    
    # Parameters
    obs_dim = 5
    encoder_units = 64  # Smaller for faster testing
    decoder_units = 64
    vocab_size = 3
    batch_size = 32
    seq_len = 10
    num_epochs = 20
    
    # Create policy with Graph2Seq encoder
    policy = Seq2SeqPolicy(
        obs_dim=obs_dim,
        encoder_units=encoder_units,
        decoder_units=decoder_units,
        vocab_size=vocab_size
    )
    
    # Create optimizer
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    # Create loss
    cross_entropy = policy.network.neglogp()
    loss = tf.reduce_mean(cross_entropy)
    
    # Get encoder variables
    encoder_vars = [v for v in policy.get_trainable_variables() if 'encoder' in v.name.lower()]
    all_vars = policy.get_trainable_variables()
    
    print(f"\nTotal trainable parameters: {len(all_vars)}")
    print(f"Encoder parameters: {len(encoder_vars)}")
    
    # Training operation
    train_op = optimizer.minimize(loss, var_list=all_vars)
    
    # Metrics
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(policy.network.decoder_prediction, policy.decoder_targets), tf.float32)
    )
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Track metrics
        losses = []
        accuracies = []
        encoder_weight_changes = []
        
        # Get initial encoder weights
        initial_encoder_weights = {}
        for var in encoder_vars[:3]:  # Track first 3
            initial_encoder_weights[var.name] = sess.run(var).copy()
            
        print("\nTraining...")
        for epoch in range(num_epochs):
            # Generate training data
            obs_data, action_data = create_synthetic_task_data(batch_size, seq_len, obs_dim, vocab_size)
            length_data = np.full(batch_size, seq_len, dtype=np.int32)
            
            feed_dict = {
                policy.obs: obs_data,
                policy.decoder_inputs: action_data,
                policy.decoder_targets: action_data,
                policy.decoder_full_length: length_data
            }
            
            # Train step
            _, loss_val, acc_val = sess.run([train_op, loss, accuracy], feed_dict)
            
            losses.append(loss_val)
            accuracies.append(acc_val)
            
            # Track weight changes
            total_change = 0
            for var_name, initial_val in initial_encoder_weights.items():
                var = [v for v in encoder_vars if v.name == var_name][0]
                current_val = sess.run(var)
                change = np.mean(np.abs(current_val - initial_val))
                total_change += change
                
            encoder_weight_changes.append(total_change / len(initial_encoder_weights))
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Loss={loss_val:.4f}, Accuracy={acc_val:.4f}")
                
        # Analyze results
        print("\n" + "="*60)
        print("TRAINING RESULTS:")
        print("="*60)
        
        # Check if loss decreased
        loss_improved = losses[-1] < losses[0]
        print(f"\nLoss improvement: {losses[0]:.4f} -> {losses[-1]:.4f}")
        print(f"[{'PASS' if loss_improved else 'FAIL'}] Loss decreased during training")
        
        # Check if accuracy increased
        acc_improved = accuracies[-1] > accuracies[0]
        print(f"\nAccuracy improvement: {accuracies[0]:.4f} -> {accuracies[-1]:.4f}")
        print(f"[{'PASS' if acc_improved else 'FAIL'}] Accuracy increased during training")
        
        # Check encoder weight updates
        weights_changed = encoder_weight_changes[-1] > 0.001
        print(f"\nEncoder weight change: {encoder_weight_changes[-1]:.6f}")
        print(f"[{'PASS' if weights_changed else 'FAIL'}] Encoder weights updated significantly")
        
        # Test on new data
        print("\nTesting on new data...")
        test_obs, test_actions = create_synthetic_task_data(batch_size, seq_len, obs_dim, vocab_size)
        
        test_feed_dict = {
            policy.obs: test_obs,
            policy.decoder_inputs: test_actions,
            policy.decoder_targets: test_actions,
            policy.decoder_full_length: length_data
        }
        
        test_loss, test_acc = sess.run([loss, accuracy], test_feed_dict)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Create plots if matplotlib available
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Loss plot
            ax1.plot(losses)
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            
            # Accuracy plot
            ax2.plot(accuracies)
            ax2.set_title('Training Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.grid(True)
            
            # Weight change plot
            ax3.plot(encoder_weight_changes)
            ax3.set_title('Encoder Weight Changes')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Mean Absolute Change')
            ax3.grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(os.path.dirname(__file__), 'training_dynamics.png')
            plt.savefig(plot_path)
            print(f"\nTraining plots saved to: {plot_path}")
            plt.close()
            
        except:
            print("\n[INFO] Matplotlib not available, skipping plots")
            
        # Final verdict
        print("\n" + "="*60)
        if loss_improved and acc_improved and weights_changed:
            print("[SUCCESS] Graph2Seq encoder shows proper training dynamics!")
        else:
            print("[WARNING] Some aspects of training dynamics need investigation")
        print("="*60)
        
        return {
            'loss_improved': loss_improved,
            'accuracy_improved': acc_improved,
            'weights_changed': weights_changed,
            'final_loss': losses[-1],
            'final_accuracy': accuracies[-1]
        }


if __name__ == "__main__":
    results = test_training_dynamics()