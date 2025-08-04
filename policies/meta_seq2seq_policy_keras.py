"""
Seq2Seq Policy refactored as Keras Model for TF2
Maintains interface compatibility with original implementation
"""
import os
import numpy as np
import tensorflow as tf
import policies.model_helper as model_helper
from policies.graph2seq_encoder import create_graph2seq_encoder, Graph2SeqEncoder
from policies.distributions.categorical_pd import CategoricalPd
import utils as U
from utils.utils import zipsame
from compat import seq2seq as contrib_seq2seq
from compat import checkpoint as compat_checkpoint
from compat import rnn as compat_rnn


class HParams:
    """Simple HParams replacement"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Seq2SeqNetwork(tf.keras.Model):
    """Seq2Seq Network as Keras Model"""
    
    def __init__(self, hparams, vocab_size, name="seq2seq_network", **kwargs):
        super(Seq2SeqNetwork, self).__init__(name=name, **kwargs)
        
        self.encoder_hidden_unit = hparams.encoder_units
        self.decoder_hidden_unit = hparams.decoder_units
        self.is_bidencoder = hparams.is_bidencoder
        self.n_features = hparams.n_features
        self.time_major = hparams.time_major
        self.is_attention = hparams.is_attention
        self.unit_type = hparams.unit_type
        self.num_layers = hparams.num_layers
        self.num_residual_layers = hparams.num_residual_layers
        self.start_token = hparams.start_token
        self.end_token = hparams.end_token
        self.vocab_size = vocab_size
        
        # Build layers
        self._build_layers()
        
    def _build_layers(self):
        """Build all the layers"""
        # Embeddings
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=[self.n_features, self.encoder_hidden_unit],
            initializer=tf.keras.initializers.RandomUniform(-1.0, 1.0),
            dtype=tf.float32,
            trainable=True
        )
        
        # Encoder embedding layer  
        self.encoder_embedding_layer = tf.keras.layers.Dense(
            self.encoder_hidden_unit,
            activation=None,
            name="encoder_embeddings"
        )
        
        # Graph2Seq encoder
        self.graph2seq_encoder = Graph2SeqEncoder(
            hidden_dim=self.encoder_hidden_unit,
            num_layers=self.num_layers,
            bidirectional=self.is_bidencoder,
            name="graph2seq_encoder"
        )
        
        # Decoder cell
        self.decoder_cell = self._build_decoder_cell()
        
        # Output projection layer
        self.output_layer = tf.keras.layers.Dense(
            self.vocab_size,
            use_bias=False,
            name="output_projection"
        )
        
        # Value function layers
        self.vf_fc1 = tf.keras.layers.Dense(128, activation='relu', name='vf_fc1')
        self.vf_fc2 = tf.keras.layers.Dense(64, activation='relu', name='vf_fc2') 
        self.vf_fc3 = tf.keras.layers.Dense(1, activation=None, name='vf_fc3')
        
        # Attention mechanism (if used)
        self.attention_mechanism = None
        self.attention_wrapper = None
        
    def _build_decoder_cell(self):
        """Build decoder RNN cell"""
        # Create RNN cells
        cells = []
        for i in range(self.num_layers):
            residual = (i >= self.num_layers - self.num_residual_layers)
            
            if self.unit_type == "lstm":
                cell = compat_rnn.BasicLSTMCell(
                    self.decoder_hidden_unit,
                    forget_bias=1.0
                )
            elif self.unit_type == "gru":
                cell = compat_rnn.GRUCell(self.decoder_hidden_unit)
            else:
                raise ValueError(f"Unknown unit type {self.unit_type}")
                
            # Add dropout wrapper if needed
            
            # Add residual wrapper if needed
            if residual and i > 0:
                cell = compat_rnn.ResidualWrapper(cell)
                
            cells.append(cell)
            
        if len(cells) == 1:
            return cells[0]
        else:
            return compat_rnn.MultiRNNCell(cells)
    
    def encode(self, encoder_inputs, training=None):
        """Run encoder"""
        # Apply encoder embeddings
        encoder_embeddings = self.encoder_embedding_layer(encoder_inputs)
        
        # Run Graph2Seq encoder
        encoder_outputs, encoder_state = self.graph2seq_encoder(
            encoder_embeddings, 
            training=training
        )
        
        return encoder_outputs, encoder_state
        
    def decode(self, decoder_inputs, encoder_outputs, encoder_state, 
               decoder_full_length, mode="train", training=None):
        """Run decoder with specified mode"""
        
        batch_size = tf.shape(decoder_inputs)[0] if decoder_inputs is not None else tf.shape(encoder_outputs)[0]
        
        # Get decoder embeddings
        if mode == "train":
            decoder_embeddings = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)
        else:
            decoder_embeddings = None
            
        # Setup helper based on mode
        if mode == "greedy":
            helper = contrib_seq2seq.GreedyEmbeddingHelper(
                self.embeddings,
                start_tokens=tf.fill([batch_size], self.start_token),
                end_token=self.end_token
            )
        elif mode == "sample":
            helper = FixedSequenceLearningSampleEmbeddingHelper(
                sequence_length=decoder_full_length,
                embedding=self.embeddings,
                start_tokens=tf.fill([batch_size], self.start_token),
                end_token=self.end_token
            )
        else:  # train mode
            helper = contrib_seq2seq.TrainingHelper(
                decoder_embeddings,
                decoder_full_length,
                time_major=self.time_major
            )
            
        # Setup decoder cell with attention if needed
        if self.is_attention:
            # Setup attention
            attention_states = encoder_outputs if not self.time_major else tf.transpose(encoder_outputs, [1, 0, 2])
            
            if self.attention_mechanism is None:
                self.attention_mechanism = contrib_seq2seq.LuongAttention(
                    self.decoder_hidden_unit, 
                    attention_states
                )
                
            if self.attention_wrapper is None:
                self.attention_wrapper = contrib_seq2seq.AttentionWrapper(
                    self.decoder_cell,
                    self.attention_mechanism,
                    attention_layer_size=self.decoder_hidden_unit
                )
                
            decoder_cell = self.attention_wrapper
            
            # Create initial state with attention
            decoder_initial_state = decoder_cell.zero_state(
                batch_size, dtype=tf.float32
            ).clone(cell_state=encoder_state)
        else:
            decoder_cell = self.decoder_cell
            decoder_initial_state = encoder_state
            
        # Create decoder
        decoder = contrib_seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=helper,
            initial_state=decoder_initial_state,
            output_layer=self.output_layer
        )
        
        # Dynamic decode
        max_iterations = decoder_full_length[0] if decoder_full_length is not None else None
        outputs, final_state, _ = contrib_seq2seq.dynamic_decode(
            decoder,
            output_time_major=self.time_major,
            maximum_iterations=max_iterations
        )
        
        return outputs, final_state
        
    def compute_value_function(self, encoder_state):
        """Compute value function from encoder state"""
        # Extract state from LSTM tuple if needed
        if isinstance(encoder_state, tuple):
            # For multi-layer, use the last layer's hidden state
            if isinstance(encoder_state[0], tuple):
                state = encoder_state[-1][1]  # Last layer's h
            else:
                state = encoder_state[1]  # Single layer's h
        else:
            state = encoder_state
            
        # Pass through value function layers
        vf = self.vf_fc1(state)
        vf = self.vf_fc2(vf)
        vf = self.vf_fc3(vf)
        
        return tf.squeeze(vf, axis=-1)
        
    def call(self, inputs, training=None):
        """Forward pass through the network
        
        Args:
            inputs: dict with keys:
                - encoder_inputs: [batch, seq_len, input_dim]
                - decoder_inputs: [batch, seq_len] (for training)
                - decoder_targets: [batch, seq_len] (for training)
                - decoder_full_length: [batch]
                - mode: "train", "greedy", or "sample"
        """
        encoder_inputs = inputs['encoder_inputs']
        decoder_inputs = inputs.get('decoder_inputs', None)
        decoder_full_length = inputs['decoder_full_length']
        mode = inputs.get('mode', 'train')
        
        # Encode
        encoder_outputs, encoder_state = self.encode(encoder_inputs, training=training)
        
        # Decode
        decoder_outputs, _ = self.decode(
            decoder_inputs, encoder_outputs, encoder_state,
            decoder_full_length, mode=mode, training=training
        )
        
        # Compute value function
        value_function = self.compute_value_function(encoder_state)
        
        # Extract logits
        if hasattr(decoder_outputs, 'rnn_output'):
            logits = decoder_outputs.rnn_output
        else:
            logits = decoder_outputs
            
        return {
            'logits': logits,
            'value_function': value_function,
            'encoder_outputs': encoder_outputs,
            'encoder_state': encoder_state
        }


class Seq2SeqPolicy(tf.keras.Model):
    """Seq2Seq Policy as Keras Model"""
    
    def __init__(self, obs_dim, encoder_units, decoder_units, vocab_size, name="policy", **kwargs):
        super(Seq2SeqPolicy, self).__init__(name=name, **kwargs)
        
        self.obs_dim = obs_dim
        self.action_dim = vocab_size
        
        # Create HParams
        hparams = HParams(
            unit_type="lstm",
            encoder_units=encoder_units,
            decoder_units=decoder_units,
            num_layers=1,
            num_residual_layers=0,
            time_major=False,
            is_bidencoder=False,
            is_attention=True,
            attention_option="luong",
            output_attention=True,
            encoder=None,
            n_features=vocab_size,
            start_token=0,
            end_token=1
        )
        
        # Build network
        self.network = Seq2SeqNetwork(hparams, vocab_size, name=f"{name}_network")
        
        # Distribution
        self.distribution = CategoricalPd()
        
    def call(self, inputs, training=None):
        """Forward pass
        
        Args:
            inputs: dict with encoder_inputs, decoder_inputs, etc
        Returns:
            dict with logits, value_function, actions, neglogp
        """
        # Run network
        outputs = self.network(inputs, training=training)
        
        # Setup distribution
        self.distribution.set_logits(outputs['logits'])
        
        # Sample actions if not in training mode
        if inputs.get('mode', 'train') != 'train':
            actions = self.distribution.sample()
            neglogp = self.distribution.neglogp(actions)
        else:
            actions = None
            neglogp = None
            
        return {
            'logits': outputs['logits'],
            'value_function': outputs['value_function'],
            'actions': actions,
            'neglogp': neglogp,
            'encoder_outputs': outputs['encoder_outputs'],
            'encoder_state': outputs['encoder_state']
        }
        
    def get_actions(self, observations, decoder_inputs=None, decoder_full_length=None):
        """Get actions for given observations"""
        if decoder_full_length is None:
            # Infer from observations shape
            decoder_full_length = tf.fill([tf.shape(observations)[0]], tf.shape(observations)[1])
            
        if decoder_inputs is None:
            # For inference, start with zeros
            batch_size = tf.shape(observations)[0]
            decoder_inputs = tf.zeros([batch_size, 1], dtype=tf.int32)
            
        inputs = {
            'encoder_inputs': observations,
            'decoder_inputs': decoder_inputs,
            'decoder_full_length': decoder_full_length,
            'mode': 'greedy'
        }
        
        outputs = self(inputs, training=False)
        
        return outputs['actions'], outputs['logits'], outputs['value_function']
        
    def save_variables(self, save_path, sess=None):
        """Save variables using joblib format for compatibility"""
        variables = self.trainable_variables
        compat_checkpoint.save_variables_joblib(variables, save_path)
        
    def load_variables(self, load_path, sess=None):
        """Load variables from joblib format"""
        variables = self.trainable_variables
        compat_checkpoint.load_variables_joblib(variables, load_path)


class FixedSequenceLearningSampleEmbeddingHelper(contrib_seq2seq.SampleEmbeddingHelper):
    """Helper for fixed sequence length sampling"""
    
    def __init__(self, sequence_length, embedding, start_tokens, end_token, 
                 softmax_temperature=None, seed=None):
        super().__init__(embedding, start_tokens, end_token, softmax_temperature, seed)
        self._sequence_length = tf.convert_to_tensor(sequence_length, name="sequence_length")
        if len(self._sequence_length.shape) != 1:
            raise ValueError(
                f"Expected sequence_length to be a vector, but received shape: {self._sequence_length.shape}"
            )
            
    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Get next inputs - override to use fixed sequence length"""
        del outputs  # unused
        
        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = tf.reduce_all(finished)
        
        next_inputs = tf.cond(
            all_finished,
            lambda: tf.zeros_like(self._embedding_fn(sample_ids)),
            lambda: self._embedding_fn(sample_ids)
        )
        
        return (finished, next_inputs, state)


class MetaSeq2SeqPolicy(tf.keras.Model):
    """Meta Seq2Seq Policy managing multiple task policies"""
    
    def __init__(self, meta_batch_size, obs_dim, encoder_units, decoder_units, vocab_size, **kwargs):
        super(MetaSeq2SeqPolicy, self).__init__(**kwargs)
        
        self.meta_batch_size = meta_batch_size
        self.action_dim = vocab_size
        
        # Create core policy
        self.core_policy = Seq2SeqPolicy(
            obs_dim, encoder_units, decoder_units, vocab_size, 
            name="core_policy"
        )
        
        # Create meta policies for each task
        self.meta_policies = []
        for i in range(meta_batch_size):
            policy = Seq2SeqPolicy(
                obs_dim, encoder_units, decoder_units, vocab_size,
                name=f"task_{i}_policy"
            )
            self.meta_policies.append(policy)
            
        # Distribution (shared)
        self.distribution = CategoricalPd()
        
    def async_parameters(self):
        """Sync parameters from core policy to task policies"""
        core_vars = self.core_policy.trainable_variables
        
        for task_policy in self.meta_policies:
            task_vars = task_policy.trainable_variables
            # Copy weights
            for core_var, task_var in zip(core_vars, task_vars):
                task_var.assign(core_var)
                
    def get_all_variables(self):
        """Get all variables from all policies"""
        all_vars = list(self.core_policy.trainable_variables)
        for policy in self.meta_policies:
            all_vars.extend(policy.trainable_variables)
        return all_vars