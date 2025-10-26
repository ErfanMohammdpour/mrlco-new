import numpy as np
import tensorflow as tf
import utils.logger as logger

tf.get_logger().setLevel('WARNING')

def _single_cell(unit_type, num_units, forget_bias, dropout, mode,
                 residual_connection=False, device_str=None, residual_fn=None):
  """Create an instance of a single RNN cell."""
  # dropout (= 1 - keep_prob) is set to 0 during eval and infer
  dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

  # Cell Type
  if unit_type == "lstm":
    single_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units,
        forget_bias=forget_bias)
  elif unit_type == "gru":
    single_cell = tf.contrib.rnn.GRUCell(num_units)
  elif unit_type == "layer_norm_lstm":
    single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        num_units,
        forget_bias=forget_bias,
        layer_norm=True)
  elif unit_type == "nas":
    single_cell = tf.contrib.rnn.NASCell(num_units)
  else:
    raise ValueError("Unknown unit type %s!" % unit_type)

  if dropout > 0.0:
    single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))

  # Residual
  if residual_connection:
    single_cell = tf.contrib.rnn.ResidualWrapper(
        single_cell, residual_fn=residual_fn)

  return single_cell


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, num_gpus, base_gpu=0,
               single_cell_fn=None, residual_fn=None):
  if not single_cell_fn:
    single_cell_fn = _single_cell

  cell_list = []
  for i in range(num_layers):
    single_cell = single_cell_fn(
        unit_type=unit_type,
        num_units=num_units,
        forget_bias=forget_bias,
        dropout=dropout,
        mode=mode,
        residual_connection=(i >= num_layers - num_residual_layers),
        residual_fn=residual_fn
    )
    cell_list.append(single_cell)

  return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, num_gpus, base_gpu=0,
                    single_cell_fn=None):

  cell_list = _cell_list(unit_type=unit_type,
                         num_units=num_units,
                         num_layers=num_layers,
                         num_residual_layers=num_residual_layers,
                         forget_bias=forget_bias,
                         dropout=dropout,
                         mode=mode,
                         num_gpus=num_gpus,
                         base_gpu=base_gpu,
                         single_cell_fn=single_cell_fn)

  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  else:  # Multi layers
    return tf.contrib.rnn.MultiRNNCell(cell_list)




# ============================================================
# ===   Graph2Seq Forward + FOMAML Meta‑Update              ===
# ============================================================
from policies.graph2seq_encoder import create_graph2seq_encoder


def make_custom_getter(theta_dict):
    """Ensure all variable lookups come from theta_dict when building symbolic nets."""
    def getter(getter, name, *args, **kwargs):
        if name in theta_dict:
            return theta_dict[name]
        return getter(name, *args, **kwargs)
    return getter


def build_seq2seq_forward_pass(encoder_inputs, decoder_inputs, decoder_targets,
                               decoder_full_length, vocab_size, hparams, theta_vars_dict):
    """Functional forward pass for the Seq2Seq network using Graph2Seq encoder."""
    with tf.compat.v1.variable_scope("seq2seq_forward", reuse=tf.compat.v1.AUTO_REUSE,
                                     custom_getter=make_custom_getter(theta_vars_dict)):
        embeddings = tf.compat.v1.get_variable("embeddings",
            [vocab_size, hparams.encoder_units],
            initializer=tf.random_uniform_initializer(-1.0, 1.0))
        encoder_embedded = tf.compat.v1.layers.dense(
            encoder_inputs, hparams.encoder_units, activation=None, name="encoder_embedding")
        decoder_embedded_inputs = tf.nn.embedding_lookup(embeddings, decoder_inputs)
        decoder_target_onehot = tf.one_hot(decoder_targets, vocab_size, dtype=tf.float32)

        encoder_outputs, encoder_state = create_graph2seq_encoder(
            encoder_inputs=encoder_embedded,
            encoder_units=hparams.encoder_units,
            num_layers=hparams.num_layers,
            is_bidirectional=hparams.is_bidencoder,
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            scope_name="encoder")

        decoder_cell = create_rnn_cell(hparams.unit_type, hparams.decoder_units,
                                       hparams.num_layers, hparams.num_residual_layers,
                                       hparams.forget_bias, hparams.dropout,
                                       tf.contrib.learn.ModeKeys.TRAIN, hparams.num_gpus)

        if hparams.is_attention:
            attn_mech = tf.contrib.seq2seq.LuongAttention(hparams.decoder_units, memory=encoder_outputs)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attn_mech, attention_layer_size=hparams.decoder_units)
            decoder_initial_state = decoder_cell.zero_state(
                tf.shape(encoder_inputs)[0], dtype=tf.float32).clone(cell_state=encoder_state)
        else:
            decoder_initial_state = encoder_state

        output_layer = tf.compat.v1.layers.Dense(vocab_size, use_bias=False, name="output_projection")

        helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedded_inputs, decoder_full_length, time_major=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell, helper=helper,
            initial_state=decoder_initial_state, output_layer=output_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, output_time_major=False,
            maximum_iterations=tf.reduce_max(decoder_full_length))

        logits = outputs.rnn_output
        pi = tf.nn.softmax(logits)
        q = tf.compat.v1.layers.dense(logits, vocab_size, activation=None, name="qvalue_layer")
        vf = tf.reduce_sum(pi * q, axis=-1)

        loss_per_timestep = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=decoder_target_onehot, logits=logits)
        loss = tf.reduce_mean(loss_per_timestep)

        return {'loss': loss, 'logits': logits, 'vf': vf}


# ============================================================
# ===        FOMAML Meta‑Update Logic                       ===
# ============================================================
def maml_batch_meta_update_fomaml(tasks_inputs,
                                  hparams,
                                  theta_vars,
                                  inner_lr,
                                  num_inner_steps,
                                  vocab_size):
    """Implements symbolic First‑Order MAML update (no 2nd‑order gradients)."""
    meta_losses = []
    meta_grads_by_task = []

    def forward_fn(inputs, var_dict):
        return build_seq2seq_forward_pass(
            encoder_inputs=inputs['encoder_inputs'],
            decoder_inputs=inputs['decoder_inputs'],
            decoder_targets=inputs['decoder_targets'],
            decoder_full_length=inputs['decoder_full_length'],
            vocab_size=vocab_size,
            hparams=hparams,
            theta_vars_dict=var_dict)

    for i, input_data in enumerate(tasks_inputs):
        # ---- Inner Update (First‑Order, stop_gradient) ----
        theta_prime = theta_vars
        for step in range(num_inner_steps):
            theta_dict = {v.name.split(":")[0]: v for v in theta_prime}
            output = forward_fn(input_data, theta_dict)
            loss = output['loss']
            grads = tf.gradients(loss, theta_prime)
            grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, theta_prime)]
            theta_prime = [tf.stop_gradient(w - inner_lr * g) for w, g in zip(theta_prime, grads)]

        # ---- Outer Loss (grad w.r.t. initial theta_vars) ----
        theta_dict = {v.name.split(":")[0]: v for v in theta_prime}
        output = forward_fn(input_data, theta_dict)
        meta_loss = output['loss']
        raw_grads = tf.gradients(meta_loss, theta_vars)
        meta_grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(raw_grads, theta_vars)]

        meta_losses.append(meta_loss)
        meta_grads_by_task.append(meta_grads)

    # ---- Average Gradients Across Tasks ----
    meta_grads_stacked = list(zip(*meta_grads_by_task))
    avg_grads = [tf.reduce_mean(tf.stack(gs), axis=0) for gs in meta_grads_stacked]

    # ---- Apply Outer Update ----
    with tf.compat.v1.variable_scope("fomaml_outer_optimizer", reuse=tf.compat.v1.AUTO_REUSE):
        optimizer = tf.compat.v1.train.AdamOptimizer()
        apply_op = optimizer.apply_gradients(zip(avg_grads, theta_vars))

        # ensure optimizer slot vars get initialized
        opt_slots = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=optimizer._name)
        for v in opt_slots:
            tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, v)

    return apply_op, meta_losses