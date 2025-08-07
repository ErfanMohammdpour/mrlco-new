#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import sys
import time
from utils import logger

print(f"TensorFlow version: {tf.__version__}")

# Disable GPU for faster testing
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Minimal configuration for testing
META_BATCH_SIZE = 1
BATCH_SIZE = 1
GRAPH_NUMBER = 1
N_ITERATIONS = 2  # Run 2 iterations to prove it works

# Configure logger
tf.get_logger().setLevel('ERROR')
logger.configure(dir="./test_iterations_log/", format_strs=['stdout'])

print(f"Configuration:")
print(f"  META_BATCH_SIZE: {META_BATCH_SIZE}")
print(f"  BATCH_SIZE: {BATCH_SIZE}")
print(f"  GRAPH_NUMBER: {GRAPH_NUMBER}")
print(f"  N_ITERATIONS: {N_ITERATIONS}")
print()

# Import components
from env.mec_offloading_envs.offloading_env import Resources, OffloadingEnvironment
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
from baselines.vf_baseline import ValueFunctionBaseline
from meta_algos.MRLCO import MRLCO

class Trainer(object):
    def __init__(self, algo, env, sampler, sample_processor, policy, n_itr, 
                 greedy_finish_time, start_itr=0, inner_batch_size=500):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sampler_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.inner_batch_size = inner_batch_size
        self.greedy_finish_time = greedy_finish_time

    def train(self):
        """Simplified training loop for testing"""
        avg_ret = []
        avg_loss = []
        
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            task_ids = self.sampler.update_tasks()
            logger.log(f"Task IDs: {task_ids}")
            
            logger.log("Obtaining samples...")
            paths = self.sampler.obtain_samples(log=False, log_prefix='')
            logger.log(f"Got {len(paths)} paths")

            greedy_run_time = [self.greedy_finish_time[x] for x in task_ids]
            logger.logkv('Average greedy latency', np.mean(greedy_run_time))

            logger.log("Processing samples...")
            samples_data = self.sampler_processor.process_samples(paths, log=False, log_prefix='')

            logger.log("Running inner policy update...")
            policy_losses, value_losses = self.algo.UpdatePPOTarget(samples_data, batch_size=self.inner_batch_size)
            
            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)
            logger.log(f"Average policy loss: {avg_policy_loss}")
            logger.log(f"Average value loss: {avg_value_loss}")
            avg_loss.append(avg_policy_loss)

            logger.log("Evaluate the one-step update for sub-task policy")
            new_paths = self.sampler.obtain_samples(log=True, log_prefix='')
            new_samples_data = self.sampler_processor.process_samples(new_paths, log="all", log_prefix='')

            logger.log("Optimizing meta policy...")
            self.algo.UpdateMetaPolicy()

            # Calculate average reward
            ret = np.array([])
            num_samples = min(5, len(new_samples_data))
            for i in range(num_samples):
                ret = np.concatenate((ret, np.sum(new_samples_data[i]['rewards'], axis=-1)), axis=-1)
            avg_reward = np.mean(ret) if len(ret) > 0 else 0.0

            logger.logkv('Itr', itr)
            logger.logkv('Average reward', avg_reward)
            logger.logkv('Iteration time', time.time() - itr_start_time)
            logger.dumpkvs()
            
            avg_ret.append(avg_reward)
            
            logger.log(f"✅ Iteration {itr} completed successfully!")

        return avg_ret, avg_loss

if __name__ == "__main__":
    print("Creating environment...")
    resource_cluster = Resources(
        mec_process_capable=(10.0 * 1024 * 1024),
        mobile_process_capable=(1.0 * 1024 * 1024),
        bandwidth_up=7.0, 
        bandwidth_dl=7.0
    )

    env = OffloadingEnvironment(
        resource_cluster=resource_cluster,
        batch_size=BATCH_SIZE,
        graph_number=GRAPH_NUMBER,
        graph_file_paths=[
            "./env/mec_offloading_envs/data/meta_offloading_20/offload_random20_1/random.20.",
        ],
        time_major=False
    )
    
    print("Getting greedy solution...")
    action, greedy_finish_time = env.greedy_solution()
    print(f"Average greedy solution: {np.mean(greedy_finish_time)}")
    
    print("Creating baseline...")
    baseline = ValueFunctionBaseline()
    
    print("Creating policy...")
    meta_policy = MetaSeq2SeqPolicy(
        meta_batch_size=META_BATCH_SIZE, 
        obs_dim=17, 
        encoder_units=128, 
        decoder_units=128,
        vocab_size=2
    )
    
    print("Creating sampler...")
    sampler = Seq2SeqMetaSampler(
        env=env,
        policy=meta_policy,
        rollouts_per_meta_task=1,
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=20000,
        parallel=False,
    )
    
    print("Creating sample processor...")
    sample_processor = Seq2SeqMetaSamplerProcessor(
        baseline=baseline,
        discount=0.99,
        gae_lambda=0.95,
        normalize_adv=True,
        positive_adv=False
    )
    
    print("Creating MRLCO algorithm...")
    algo = MRLCO(
        policy=meta_policy,
        meta_batch_size=META_BATCH_SIZE
    )
    
    print("Creating trainer...")
    trainer = Trainer(
        algo=algo,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        policy=meta_policy,
        n_itr=N_ITERATIONS,
        greedy_finish_time=greedy_finish_time,
        start_itr=0,
        inner_batch_size=500
    )
    
    print("\n" + "="*60)
    print("Starting training loop...")
    print("="*60)
    
    start_time = time.time()
    avg_ret, avg_loss = trainer.train()
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ Training completed successfully!")
    print("="*60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average rewards: {avg_ret}")
    print(f"Average losses: {avg_loss}")