import os
import tensorflow as tf
# Disable eager execution for TF1 compatibility
tf.compat.v1.disable_eager_execution()
import numpy as np
import time
from utils import logger
from scripts.automated_reporting import create_training_report

# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_gpu_memory_growth():
    """Enable memory growth for all visible GPUs"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s). Enabling memory growth...")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  - Memory growth enabled for {gpu}")
            except RuntimeError as e:
                print(f"  - Failed to enable memory growth for {gpu}: {e}")
    return gpus

def detect_and_configure_devices():
    """Detect available devices and configure TensorFlow accordingly"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    num_gpus = len(gpus)
    
    print(f"\n=== Device Configuration ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Visible GPUs: {num_gpus}")
    
    if num_gpus > 0:
        print(f"Will use GPU device(s): {[gpu.name for gpu in gpus]}")
        device_name = '/GPU:0'  # Default to first GPU
    else:
        print("No GPUs available. Will use CPU.")
        device_name = '/CPU:0'
    
    print("===========================\n")
    
    return num_gpus, device_name

def warmup_device(device_name, sess):
    """Run a simple matmul to verify device is working"""
    print(f"Running device warmup on {device_name}...")
    
    with tf.device(device_name):
        # Small matmul operation
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
    
    result = sess.run(c)
    print(f"Warmup matmul result: \n{result}")
    print(f"Device warmup complete on {device_name}.\n")

if __name__ == "__main__":
    print("Starting imports...")
    from env.mec_offloaing_envs.offloading_env import Resources
    from env.mec_offloaing_envs.offloading_env import OffloadingEnvironment
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
    from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
    from baselines.vf_baseline import ValueFunctionBaseline
    from meta_algos.MRLCO import MRLCO
    print("Imports complete")

    # Setup GPU memory growth first
    gpus = setup_gpu_memory_growth()
    
    # Detect and configure devices
    num_gpus, device_name = detect_and_configure_devices()
    
    print("Setting TF logging...")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logger.configure(dir="./meta_offloading20_log-inner_step1/", format_strs=['stdout', 'log', 'csv'])

    META_BATCH_SIZE = 2  # Reduced for testing
    
    print("Creating resource cluster...")
    resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024),
                                 bandwidth_up=7.0, bandwidth_dl=7.0)
    print("Creating environment...")
    # Use smaller dataset for testing
    env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                batch_size=10,  # Reduced from 100
                                graph_number=10,  # Reduced from 100
                                graph_file_paths=[
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_1/random.20.",
                                    "./env/mec_offloaing_envs/data/meta_offloading_20/offload_random20_2/random.20.",
                                ],
                                time_major=False)

    action, greedy_finish_time = env.greedy_solution()
    print("avg greedy solution: ", np.mean(greedy_finish_time))
    
    baseline = ValueFunctionBaseline()

    meta_policy = MetaSeq2SeqPolicy(meta_batch_size=META_BATCH_SIZE, obs_dim=17, encoder_units=32, decoder_units=32,
                                    vocab_size=2)

    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=200,  # Reduced from 20000
        parallel=False,
    )

    sample_processor = Seq2SeqMetaSamplerProcessor(baseline=baseline,
                                                   discount=0.99,
                                                   gae_lambda=0.95,
                                                   normalize_adv=True,
                                                   positive_adv=False)
    algo = MRLCO(policy=meta_policy,
                         meta_sampler=sampler,
                         meta_sampler_process=sample_processor,
                         inner_lr=5e-4,
                         outer_lr=5e-4,
                         meta_batch_size=META_BATCH_SIZE,
                         num_inner_grad_steps=1,
                         clip_value = 0.3)

    # Create session with proper config
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True  # IMPORTANT: Allow TF to use CPU if GPU ops fail
    config.log_device_placement = False
    
    with tf.compat.v1.Session(config=config) as sess:
        # Run warmup after session creation but before variable initialization
        if num_gpus > 0:
            try:
                warmup_device(device_name, sess)
            except Exception as e:
                print(f"Warning: Device warmup failed: {e}")
                print("Continuing with training...")
        
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Run just one iteration
        print("\n============ Running ONE iteration ============")
        itr = 0
        logger.log("\n ---------------- Iteration %d ----------------" % itr)
        logger.log("Sampling set of tasks/goals for this meta-batch...")

        task_ids = sampler.update_tasks()
        paths = sampler.obtain_samples(log=False, log_prefix='')

        greedy_run_time = [greedy_finish_time[x] for x in task_ids]
        logger.logkv('Average greedy latency,', np.mean(greedy_run_time))

        logger.log("Processing samples...")
        samples_data = sample_processor.process_samples(paths, log=False, log_prefix='')

        policy_losses, value_losses = algo.UpdatePPOTarget(samples_data, batch_size=100)

        print("average task losses: ", np.mean(policy_losses))
        print("average value losses: ", np.mean(value_losses))

        print("Evaluate the one-step update for sub-task policy")
        new_paths = sampler.obtain_samples(log=True, log_prefix='')
        new_samples_data = sample_processor.process_samples(new_paths, log="all", log_prefix='')

        logger.log("Optimizing policy...")
        algo.UpdateMetaPolicy()

        ret = np.array([])
        for i in range(len(new_samples_data)):
            ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)

        avg_reward = np.mean(ret)

        latency = np.array([])
        for i in range(len(new_samples_data)):
            latency = np.concatenate((latency, new_samples_data[i]['finish_time']), axis=-1)

        avg_latency = np.mean(latency)

        logger.logkv('Itr', itr)
        logger.logkv('Average reward, ', avg_reward)
        logger.logkv('Average latency,', avg_latency)

        logger.dumpkvs()
        
        print("\n============ ONE iteration completed successfully! ============")