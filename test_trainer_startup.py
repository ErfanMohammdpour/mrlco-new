import os
import sys
import tensorflow as tf

# Disable eager execution for TF1 compatibility
tf.compat.v1.disable_eager_execution()

# Set up environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN to reduce warnings

print("Starting trainer test...")
print("TensorFlow version:", tf.__version__)
print("Python version:", sys.version)

# Test with CPU first
os.environ['CUDA_VISIBLE_DEVICES'] = ''
print("\n=== Testing with CPU only ===")

# Basic imports
print("Testing imports...")
try:
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    print("Imports successful!")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

print("\nCreating small test environment...")
resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024),
                             bandwidth_up=7.0, bandwidth_dl=7.0)

# Use smaller batch for testing
env = OffloadingEnvironment(resource_cluster=resource_cluster,
                            batch_size=10,  # Reduced from 100
                            graph_number=10,  # Reduced from 100
                            graph_file_paths=[
                                "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                            ],
                            time_major=False)

print("Environment created successfully!")

# Test policy creation
print("\nCreating policy...")
meta_policy = MetaSeq2SeqPolicy(meta_batch_size=2, obs_dim=17, encoder_units=32, decoder_units=32,
                                vocab_size=2)
print("Policy created successfully!")

# Create session and test
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

print("\nTesting session creation...")
with tf.compat.v1.Session(config=config) as sess:
    print("Session created!")
    sess.run(tf.compat.v1.global_variables_initializer())
    print("Variables initialized!")
    
print("\nStartup test completed successfully!")