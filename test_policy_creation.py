import tensorflow as tf
# Disable eager execution
tf.compat.v1.disable_eager_execution()

print("Testing policy creation...")

try:
    from policies.meta_seq2seq_policy import MetaSeq2SeqPolicy
    print("✓ Imported MetaSeq2SeqPolicy")
    
    print("\nCreating MetaSeq2SeqPolicy...")
    meta_policy = MetaSeq2SeqPolicy(
        meta_batch_size=2,  # Small for testing
        obs_dim=17, 
        encoder_units=128, 
        decoder_units=128,
        vocab_size=2
    )
    print("✓ MetaSeq2SeqPolicy created successfully!")
    
except Exception as e:
    print(f"✗ Failed to create MetaSeq2SeqPolicy: {e}")
    import traceback
    traceback.print_exc()
    
    # Let's test just creating the network without attention
    print("\n\nTesting without attention...")
    try:
        # Modify the policy to not use attention
        import policies.meta_seq2seq_policy as policy_module
        
        # Create a simple test
        print("Creating simple Seq2SeqPolicy without attention...")
        from policies.meta_seq2seq_policy import Seq2SeqPolicy
        
        # We'll need to patch the HParams to disable attention
        class TestSeq2SeqPolicy(Seq2SeqPolicy):
            def __init__(self, obs_dim, encoder_units, decoder_units, vocab_size, name="pi"):
                self.decoder_targets = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_targets_ph_"+name)
                self.decoder_inputs = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_inputs_ph"+name)
                self.obs = tf.compat.v1.placeholder(shape=[None, None, obs_dim], dtype=tf.float32, name="obs_ph"+name)
                self.decoder_full_length = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32, name="decoder_full_length"+name)
                
                self.action_dim = vocab_size
                self.name = name
                
                # Create HParams with attention disabled
                class HParams:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)
                
                hparams = HParams(
                    unit_type="lstm",
                    encoder_units=encoder_units,
                    decoder_units=decoder_units,
                    n_features=vocab_size,
                    time_major=False,
                    is_attention=False,  # Disable attention for testing
                    forget_bias=1.0,
                    dropout=0,
                    num_gpus=1,
                    num_layers=2,
                    num_residual_layers=0,
                    start_token=0,
                    end_token=2,
                    is_bidencoder=False
                )
                
                from policies.meta_seq2seq_policy import Seq2SeqNetwork
                self.network = Seq2SeqNetwork(
                    hparams=hparams, 
                    reuse=tf.compat.v1.AUTO_REUSE,
                    encoder_inputs=self.obs,
                    decoder_inputs=self.decoder_inputs,
                    decoder_full_length=self.decoder_full_length,
                    decoder_targets=self.decoder_targets,
                    name=name
                )
                
                self.vf = self.network.vf
                from policies.distributions.categorical_pd import CategoricalPd
                self._dist = CategoricalPd(vocab_size)
        
        test_policy = TestSeq2SeqPolicy(obs_dim=17, encoder_units=128, decoder_units=128, vocab_size=2)
        print("✓ Simple policy without attention created successfully!")
        
    except Exception as e2:
        print(f"✗ Failed to create simple policy: {e2}")
        traceback.print_exc()