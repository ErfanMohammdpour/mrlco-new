import tensorflow as tf
import numpy as np
from policies.distributions.base import Distribution

class CategoricalPd(Distribution):
    """
        General methods for a diagonal gaussian distribution of this size
        """

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Computes the symbolic representation of the KL divergence of two multivariate
        Gaussian distribution with diagonal covariance matrices

        Args:
            old_dist_info_vars (dict) : dict of old distribution parameters as tf.Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as tf.Tensor

        Returns:
            (tf.Tensor) : Symbolic representation of kl divergence (tensorflow op)
        """

        # assert ranks

        old_logits = old_dist_info_vars
        new_logits = new_dist_info_vars

        a0 = old_logits - tf.reduce_max(old_logits, axis=-1, keepdims=True)
        a1 = new_logits - tf.reduce_max(new_logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices

       Args:
            old_dist_info (dict): dict of old distribution parameters as numpy array
            new_dist_info (dict): dict of new distribution parameters as numpy array

        Returns:
            (numpy array): kl divergence of distributions
        """
        old_logits = old_dist_info
        new_logits = new_dist_info

        a0 = old_logits - np.amax(old_logits, axis=-1, keepdims=True)
        a1 = new_logits - np.amax(new_logits, axis=-1, keepdims=True)
        ea0 = np.exp(a0)
        ea1 = np.exp(a1)
        z0 = np.sum(ea0, axis=-1, keepdims=True)
        z1 = np.sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return np.sum(p0 * (a0 - np.log(z0) - a1 + np.log(z1)), axis=-1)

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        """
        Symbolic likelihood ratio p_new(x)/p_old(x) of two distributions

        Args:
            x_var (tf.Tensor): variable where to evaluate the likelihood ratio p_new(x)/p_old(x)
            old_dist_info_vars (dict) : dict of old distribution parameters as tf.Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as tf.Tensor

        Returns:
            (tf.Tensor): likelihood ratio
        """
        # print(x_var)
        # print(new_dist_info_vars)
        logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)

        return tf.exp(logli_new - logli_old)

    def log_likelihood_sym(self, x_var, logits):
        """
        Symbolic log likelihood log p(x) of the distribution

        Args:
            x_var (tf.Tensor): variable where to evaluate the log likelihood
            dist_info_vars (dict) : dict of distribution parameters as tf.Tensor

        Returns:
             (numpy array): log likelihood
        """
        target = tf.one_hot(x_var,
                   self._dim,
                   dtype=tf.float32)

        neg_log = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits,
            labels=target)

        return -neg_log

    def log_likelihood(self, xs, logits):
        """Numerically stable log p(x) for the NumPy path: `log_softmax(logits)[x]`.

        The audited implementation had three defects (F21): it normalised with the Python builtin
        `sum` (wrong axis for batched `[B, C]` logits), it took `log(one_hot * p)` which includes
        `log(0) = -inf` for the unselected classes so EVERY entry collapsed to `-inf`, and it
        built `final_shape` from `list.append` (which returns None) so the reshape was a no-op.
        It returned `[-inf, -inf]` for zero logits and actions [0, 1] instead of `-log(3)`.

        Computed as `logits - logsumexp(logits)` (a stable log-softmax) and gathered at the
        selected actions, so scalar / batched / time-indexed inputs and large logits all agree
        with the TensorFlow `log_likelihood_sym` path.

        Args:
            xs: array of action indices, broadcast-compatible with `logits[..., 0]`
            logits: array with the action axis last, shape (..., C)

        Returns:
            (numpy array) log likelihood with shape `logits.shape[:-1]`
        """
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim < 1:
            raise ValueError("logits must have at least one dimension, got %r"
                             % (logits.shape,))
        if logits.shape[-1] != int(self._dim):
            raise ValueError("logits last axis is %d but the distribution has %d actions"
                             % (logits.shape[-1], int(self._dim)))
        logits = np.nan_to_num(logits, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        # stable log-softmax along the ACTION axis (last), never a Python builtin sum
        axis = logits.ndim - 1
        shift = np.max(logits, axis=axis, keepdims=True)
        shift = np.where(np.isfinite(shift), shift, 0.0)
        shifted = logits - shift
        lse = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)) + shift
        log_probs = logits - lse
        idx = np.asarray(xs, dtype=np.int64)
        if idx.shape != log_probs.shape[:-1]:
            try:
                idx = np.broadcast_to(idx, log_probs.shape[:-1])
            except ValueError as exc:
                raise ValueError(
                    "action shape %r is not broadcastable to the logits batch shape %r"
                    % (np.asarray(xs).shape, log_probs.shape[:-1])) from exc
        if np.any((idx < 0) | (idx >= int(self._dim))):
            raise ValueError("action index outside [0, %d): %r"
                             % (int(self._dim), np.asarray(xs)))
        gathered = np.take_along_axis(log_probs, idx[..., None], axis=axis)[..., 0]
        return gathered

    def entropy_sym(self, logits):
        """
        Symbolic entropy of the distribution

        Args:
            dist_info (dict) : dict of distribution parameters as tf.Tensor

        Returns:
            (tf.Tensor): entropy
        """

        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def entropy(self, logits):
        """
        Compute the entropy of the distribution

        Args:
            dist_info (dict) : dict of distribution parameters as numpy array

        Returns:
          (numpy array): entropy
        """

        a0 = logits - np.amax(logits, axis=-1, keepdims=True)
        ea0 = np.exp(a0)
        z0 = np.sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return np.sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self, logits):
        """
        Draws a sample from the distribution

        Args:
           dist_info (dict) : dict of distribution parameter instantiations as numpy array

        Returns:
           (obj): sample drawn from the corresponding instantiation
        """
        u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)

        return tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)

    @property
    def dist_info_specs(self):
        return [("logits", (self.dim,))]