#!/usr/bin/env python
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Python version:", sys.version)

print("1. Importing TensorFlow...")
import tensorflow as tf
print("   TensorFlow version:", tf.__version__)

print("2. Importing numpy...")
import numpy as np
print("   NumPy version:", np.__version__)

print("3. Importing utils...")
from utils import logger
print("   Logger imported")

print("4. Importing environment Resources...")
from env.mec_offloading_envs.offloading_env import Resources
print("   Resources imported")

print("5. Importing environment OffloadingEnvironment...")
from env.mec_offloading_envs.offloading_env import OffloadingEnvironment
print("   OffloadingEnvironment imported")

print("6. Creating environment...")
resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024),
                             bandwidth_up=7.0, bandwidth_dl=7.0)
print("   Resource cluster created")

env = OffloadingEnvironment(resource_cluster=resource_cluster,
                            batch_size=2,  # Very small
                            graph_number=2,  # Very small
                            graph_file_paths=[
                                "./env/mec_offloading_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                "./env/mec_offloading_envs/data/meta_offloading_20/offload_random20_2/random.20.",
                            ],
                            time_major=False)
print("   Environment created")

print("7. Getting greedy solution...")
action, greedy_finish_time = env.greedy_solution()
print("   Greedy solution computed, avg:", np.mean(greedy_finish_time))

print("8. Importing policy...")
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
print("   MetaSeq2SeqPolicy imported")

print("9. Creating policy...")
META_BATCH_SIZE = 2
meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=17, 
                                encoder_units=32, decoder_units=32, vocab_size=2)
print("   Policy created")

print("\nAll imports and basic initialization successful!")