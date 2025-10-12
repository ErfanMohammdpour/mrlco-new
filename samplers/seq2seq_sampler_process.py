from samplers.base import SampleProcessor
from utils import utils
import numpy as np

class Seq2SeqSamplerProcessor(SampleProcessor):
    def process_samples(self, paths, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths (list): A list of dicts containing path data
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (dict): Processed sample data
        """
        assert isinstance(paths, list), 'paths must be a list'
        assert self.baseline, 'baseline must be specified'

        # fits baseline, comput advantages and stack path data
        samples_data, paths = self._compute_samples_data(paths)

        # log statistics if desired
        self._log_path_stats(paths, log=log, log_prefix=log_prefix)

        return samples_data

    def _compute_samples_data(self, paths):
        assert type(paths) == list

        # 1) compute discounted rewards (returns)
        for idx, path in enumerate(paths):
            # Flatten rewards for return computation
            rewards = path["rewards"]
            if len(rewards.shape) > 1:
                rewards = rewards.flatten()
            path["returns"] = utils.discount_cumsum(rewards, self.discount)

        # 2) fit baseline estimator using the path returns and predict the return baselines
        self.baseline.fit(paths, target_key="returns")
        all_path_baselines = [self.baseline.predict(path) for path in paths]

        # 3) compute advantages and adjusted rewards
        for idx, path in enumerate(paths):
            # Flatten rewards for advantage computation
            rewards = path["rewards"]
            if len(rewards.shape) > 1:
                rewards = rewards.flatten()
            
            # Compute advantages using the flattened data
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = rewards + self.discount * path_baselines[1:] - path_baselines[:-1]
            path["advantages"] = utils.discount_cumsum(deltas, self.discount * self.gae_lambda)

        observations, actions, logits, rewards, returns, values, advantages, finish_time = self._append_path_data(paths)

        decoder_full_lengths = np.array(observations.shape[0] * [observations.shape[1]])
        # 5) if desired normalize / shift advantages
        if self.normalize_adv:
            advantages = utils.normalize_advantages(advantages)
        if self.positive_adv:
            advantages = utils.shift_advantages_to_positive(advantages)

        # 6) create samples_data object
        samples_data = dict(
            observations=observations,
            decoder_full_lengths=decoder_full_lengths,
            actions=actions,
            logits=logits,
            rewards=rewards,
            returns=returns,
            values=values,
            advantages=advantages,
            finish_time=finish_time
        )

        return samples_data, paths

    def _append_path_data(self, paths):
        # Concatenate all paths into single arrays
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        logits = np.concatenate([path["logits"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        values = np.concatenate([path["values"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        finish_time = np.concatenate([path["finish_time"] for path in paths])
        
        # Ensure logits have the correct shape for the policy
        # Policy expects (batch_size, sequence_length, vocab_size)
        print(f"Original logits shape: {logits.shape}")
        
        if len(logits.shape) == 3:
            # Logits are (batch_size, sequence_length, features)
            # We need to reshape to (batch_size, sequence_length, vocab_size)
            vocab_size = 2  # This should match the policy's vocab_size
            if logits.shape[2] == vocab_size:
                # Already correct shape
                pass
            elif logits.shape[2] % vocab_size == 0:
                # Reshape from (batch_size, sequence_length, features) to (batch_size, sequence_length, vocab_size)
                sequence_length = logits.shape[1]
                features_per_step = logits.shape[2] // vocab_size
                logits = logits.reshape(logits.shape[0], sequence_length * features_per_step, vocab_size)
            else:
                # Pad or truncate to match vocab_size
                sequence_length = logits.shape[1]
                current_features = logits.shape[2]
                if current_features > vocab_size:
                    # Truncate
                    logits = logits[:, :, :vocab_size]
                else:
                    # Pad with zeros
                    padded_logits = np.zeros((logits.shape[0], sequence_length, vocab_size))
                    padded_logits[:, :, :current_features] = logits
                    logits = padded_logits
        elif len(logits.shape) == 2:
            # If logits are flattened, reshape them
            vocab_size = 2
            if logits.shape[1] % vocab_size == 0:
                sequence_length = logits.shape[1] // vocab_size
                logits = logits.reshape(logits.shape[0], sequence_length, vocab_size)
            else:
                # Pad or truncate
                sequence_length = logits.shape[1] // vocab_size
                if sequence_length * vocab_size < logits.shape[1]:
                    padded_logits = np.zeros((logits.shape[0], sequence_length + 1, vocab_size))
                    padded_logits[:, :sequence_length, :] = logits[:, :sequence_length * vocab_size].reshape(logits.shape[0], sequence_length, vocab_size)
                    logits = padded_logits
                else:
                    logits = logits[:, :sequence_length * vocab_size].reshape(logits.shape[0], sequence_length, vocab_size)
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
        
        print(f"Final logits shape: {logits.shape}")
        
        return observations, actions, logits, rewards, returns, values, advantages, finish_time