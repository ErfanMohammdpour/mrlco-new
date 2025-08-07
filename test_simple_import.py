#!/usr/bin/env python
import sys
print("Starting imports...")
sys.stdout.flush()

print("1. Importing tensorflow...")
sys.stdout.flush()
import tensorflow as tf
print(f"   TensorFlow version: {tf.__version__}")
sys.stdout.flush()

print("2. Importing numpy...")
sys.stdout.flush()
import numpy as np
print("   NumPy imported")
sys.stdout.flush()

print("3. Importing utils logger...")
sys.stdout.flush()
from utils import logger
print("   Logger imported")
sys.stdout.flush()

print("4. Importing environment...")
sys.stdout.flush()
from env.mec_offloading_envs.offloading_env import Resources, OffloadingEnvironment
print("   Environment imported")
sys.stdout.flush()

print("5. Importing policy...")
sys.stdout.flush()
from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
print("   Policy imported")
sys.stdout.flush()

print("6. Importing sampler...")
sys.stdout.flush()
from samplers.seq2seq_meta_sampler import Seq2SeqMetaSampler
print("   Sampler imported")
sys.stdout.flush()

print("7. Importing sampler processor...")
sys.stdout.flush()
from samplers.seq2seq_meta_sampler_process import Seq2SeqMetaSamplerProcessor
print("   Sampler processor imported")
sys.stdout.flush()

print("8. Importing baseline...")
sys.stdout.flush()
from baselines.vf_baseline import ValueFunctionBaseline
print("   Baseline imported")
sys.stdout.flush()

print("9. Importing MRLCO...")
sys.stdout.flush()
from meta_algos.MRLCO import MRLCO
print("   MRLCO imported")
sys.stdout.flush()

print("\n✅ All imports successful!")
sys.stdout.flush()