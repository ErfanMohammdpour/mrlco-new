import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from policies.meta_seq2seq_policy import Seq2SeqPolicy
import numpy as np

print("Creating two identical policies...")

policy1 = Seq2SeqPolicy(obs_dim=17, encoder_units=128, decoder_units=128, vocab_size=2, name='policy1')
policy2 = Seq2SeqPolicy(obs_dim=17, encoder_units=128, decoder_units=128, vocab_size=2, name='policy2')

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    vars1 = policy1.get_variables()
    vars2 = policy2.get_variables()
    
    print(f"\nPolicy1 has {len(vars1)} variables")
    print(f"Policy2 has {len(vars2)} variables")
    
    print("\nPolicy1 variables:")
    for v in vars1:
        print(f"  {v.name}: {v.shape}")
        
    print("\nPolicy2 variables:")
    for v in vars2:
        print(f"  {v.name}: {v.shape}")
        
    # Check for differences
    names1 = set(v.name for v in vars1)
    names2 = set(v.name for v in vars2)
    
    if names1 != names2:
        print("\nDifferences found!")
        print("In policy1 but not policy2:", names1 - names2)
        print("In policy2 but not policy1:", names2 - names1)