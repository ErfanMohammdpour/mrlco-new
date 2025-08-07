import tensorflow as tf
import numpy as np
import sys

print(f"TensorFlow version: {tf.__version__}")

# Enable GPU if available but with memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

from env.mec_offloading_envs.offloading_env import Resources, OffloadingEnvironment
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline
from meta_algos.MRLCO import MRLCO
from utils import logger

print("Imports successful!")

# Disable TensorFlow logging
tf.get_logger().setLevel('ERROR')
logger.configure(dir="./test_quick_log/", format_strs=['stdout'])

# MINIMAL CONFIG FOR TESTING
META_BATCH_SIZE = 2  # Reduced from 10
BATCH_SIZE = 10      # Reduced from 100
GRAPH_NUMBER = 10    # Reduced from 100

print(f"Using META_BATCH_SIZE={META_BATCH_SIZE}, BATCH_SIZE={BATCH_SIZE}, GRAPH_NUMBER={GRAPH_NUMBER}")

print("Creating resource cluster...")
resource_cluster = Resources(
    mec_process_capable=(10.0 * 1024 * 1024),
    mobile_process_capable=(1.0 * 1024 * 1024),
    bandwidth_up=7.0, 
    bandwidth_dl=7.0
)

print("Creating environment...")
env = OffloadingEnvironment(
    resource_cluster=resource_cluster,
    batch_size=BATCH_SIZE,
    graph_number=GRAPH_NUMBER,
    graph_file_paths=[
        "./env/mec_offloading_envs/data/meta_offloading_20/offload_random20_1/random.20.",
        "./env/mec_offloading_envs/data/meta_offloading_20/offload_random20_2/random.20.",
    ],
    time_major=False
)

print("Environment created successfully!")

print("Getting greedy solution...")
action, greedy_finish_time = env.greedy_solution()
print(f"avg greedy solution: {np.mean(greedy_finish_time)}")

print("\nCreating baseline...")
baseline = ValueFunctionBaseline()

print("Creating policy...")
with tf.name_scope("seq2seq_policy"):
    policy = MetaSeq2SeqPolicy(
        obs_dim=env.observation_space.shape[0],
        vocab_size=env.action_space.n,
        use_attention=True,
        attention_option='luong',
        meta_batch_size=META_BATCH_SIZE,
        baseline=baseline
    )
    
print("Policy created successfully!")

print("Creating sampler...")
sampler = Seq2SeqMetaSampler(
    env=env, 
    policy=policy,
    rollouts_per_meta_task=5,  # Reduced
    meta_batch_size=META_BATCH_SIZE,
    max_path_length=20,
    parallel=False
)

print("Creating sampler processor...")
sample_processor = Seq2SeqMetaSamplerProcessor(
    baseline=baseline,
    discount=0.99,
    gae_lambda=0.95,
    normalize_adv=True,
    positive_adv=False
)

print("Creating MRLCO algorithm...")
algo = MRLCO(
    policy=policy,
    meta_batch_size=META_BATCH_SIZE
)

print("All components initialized successfully!")

print("\n---------------- Starting Training Loop ----------------")
# Just run 2 iterations to test
for itr in range(2):
    print(f"\n---------------- Iteration {itr} ----------------")
    print("Sampling set of tasks/goals for this meta-batch...")
    
    task_ids = sampler.update_tasks()
    print(f"Task IDs: {task_ids}")
    
    print("Obtaining samples...")
    paths = sampler.obtain_samples(log=False, log_prefix='')
    print(f"Got {len(paths)} paths")
    
    print("Processing samples...")
    samples_data = sample_processor.process_samples(paths, log=False, log_prefix='')
    
    print("Running inner policy update...")
    policy_losses, value_losses = algo.UpdatePPOTarget(samples_data, batch_size=500)
    print(f"Average policy loss: {np.mean(policy_losses)}")
    print(f"Average value loss: {np.mean(value_losses)}")
    
    print("Resampling with updated policy...")
    new_paths = sampler.obtain_samples(log=True, log_prefix='')
    new_samples_data = sample_processor.process_samples(new_paths, log="all", log_prefix='')
    
    print("Running outer policy update...")
    algo.UpdateMetaPolicy()
    
    print(f"Iteration {itr} completed successfully!")

print("\n✅ Test completed successfully - no errors!")