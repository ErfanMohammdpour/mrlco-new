from samplers.base import SampleProcessor
from utils import utils
import numpy as np


class Seq2SeSamplerProcessor(SampleProcessor):
    def process_samples(self, paths, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths_meta_batch (dict): A list of dict of lists, size: [meta_batch_size] x (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (list of dicts) : Processed sample data among the meta-batch; size: [meta_batch_size] x [7] x (batch_size x max_path_length)
        """
        assert self.baseline, 'baseline must be specified'

        all_paths = []

        # fits baseline, comput advantages and stack path data
        samples_data, paths = self._compute_samples_data(paths)

        all_paths.extend(paths)

        return samples_data


    def _compute_samples_data(self, paths):
        assert type(paths) == list

        # 1) compute discounted rewards (returns)
        for idx, path in enumerate(paths):
            path["returns"] = utils.discount_cumsum(path["rewards"], self.discount)

        # 2) fit baseline estimator using the path returns and predict the return baselines
        self.baseline.fit(paths, target_key="returns")
        all_path_baselines = [self.baseline.predict(path) for path in paths]

        # 3) compute advantages and adjusted rewards
        paths = self._compute_advantages(paths, all_path_baselines)

        path_data = self._append_path_data(paths)
        if len(path_data) == 9:  # With energy
            observations, actions, logits, rewards, returns, values, advantages, finish_time, energy = path_data
        else:  # Without energy
            observations, actions, logits, rewards, returns, values, advantages, finish_time = path_data
            energy = None

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
        
        # Add energy if available
        if energy is not None:
            samples_data['energy'] = energy

        return samples_data, paths

    def _append_path_data(self, paths):
        observations = np.array([path["observations"] for path in paths])
        actions = np.array([path["actions"] for path in paths])

        logits = np.array([path["logits"] for path in paths])
        rewards = np.array([path["rewards"] for path in paths])
        returns = np.array([path["returns"] for path in paths])
        values = np.array([path["values"] for path in paths])
        advantages = np.array([path["advantages"] for path in paths])
        
        # Handle finish_time - ensure it's a scalar for each path
        finish_times = []
        for path in paths:
            ft = path["finish_time"]
            # Convert to scalar if it's an array/list
            if isinstance(ft, (list, np.ndarray)):
                if len(ft) > 0:
                    # Take the last element (final finish time) or max if multiple
                    ft = float(np.max(ft) if len(ft) > 1 else ft[-1] if len(ft) == 1 else ft[0])
                else:
                    ft = 0.0
            finish_times.append(ft)
        finish_time = np.array(finish_times)
        
        # Handle energy if present (optional, for logging)
        if "energy" in paths[0] and paths[0]["energy"] is not None:
            energy = np.array([path["energy"] for path in paths])
            return observations, actions, logits, rewards, returns, values, advantages, finish_time, energy
        else:
            return observations, actions, logits, rewards, returns, values, advantages, finish_time
