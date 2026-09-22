from samplers.base import SampleProcessor

from env.mec_offloaing_envs.scheduler.mask_metrics import accumulate
from env.mec_offloaing_envs.scheduler.masking import resolve_mask_mode
from utils import utils
import numpy as np

class Seq2SeqMetaSamplerProcessor(SampleProcessor):
    def process_samples(self, paths_meta_batch, log=False, log_prefix=''):
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
        assert isinstance(paths_meta_batch, dict), 'paths must be a dict'
        assert self.baseline, 'baseline must be specified'

        samples_data_meta_batch = []
        all_paths = []

        for meta_task, paths in paths_meta_batch.items():

            # fits baseline, comput advantages and stack path data
            samples_data, paths = self._compute_samples_data(paths)

            samples_data_meta_batch.append(samples_data)
            all_paths.extend(paths)

        # 7) compute normalized trajectory-batch rewards (for E-MAML)
        overall_avg_reward = np.mean(
            np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))
        overall_avg_reward_std = np.std(
            np.concatenate([samples_data['rewards'] for samples_data in samples_data_meta_batch]))

        for samples_data in samples_data_meta_batch:
            samples_data['adj_avg_rewards'] = (samples_data['rewards'] - overall_avg_reward) / (
                        overall_avg_reward_std + 1e-8)

        # 8) log statistics if desired
        self._log_path_stats(all_paths, log=log, log_prefix=log_prefix)

        return samples_data_meta_batch

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
        observations = path_data["observations"]
        actions = path_data["actions"]
        logits = path_data["logits"]
        rewards = path_data["rewards"]
        returns = path_data["returns"]
        values = path_data["values"]
        advantages = path_data["advantages"]
        finish_time = path_data["finish_time"]
        energy = path_data["energy"]
        feasible = path_data["feasible"]
        raw_logits = path_data["raw_logits"]

        decoder_full_lengths = np.array(observations.shape[0] * [observations.shape[1]])
        # Diagnostic POMO: A_i = R_i - mean_graph(R). Skip global adv-norm (would mix graphs).
        if getattr(self, "pomo_elite", False):
            from spec.learning_ops import apply_pomo_token_advantages

            advantages = apply_pomo_token_advantages(
                rewards, int(getattr(self, "pomo_n_instances", 20))
            )
        elif self.normalize_adv:
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

        # Feasibility mask travels with the batch: the PPO update must use the
        # rollout mask, never a recomputation (MASKED_PPO_INTERFACE_6b).
        if feasible is not None:
            samples_data['feasible'] = feasible
        if raw_logits is not None:
            samples_data['raw_logits'] = raw_logits

        # ⑥b metrics: aggregate over TOKENS with the stored rollout mask and the
        # raw logits captured in the same sess.run. In active mode a missing or
        # mis-shaped mask raises here instead of reporting zeros.
        samples_data['mask_accumulator'] = accumulate(
            pre_guard=feasible,
            actions=actions,
            raw_logits=raw_logits,
            values=values,
            require_mask=self.resolved_mask_mode() != "off",
        )

        return samples_data, paths

    def resolved_mask_mode(self):
        """Explicit mode set by the trainer, else the process environment."""
        explicit = getattr(self, "mask_mode", None)
        return explicit if explicit is not None else resolve_mask_mode()

    def _append_path_data(self, paths):
        observations = np.array([path["observations"] for path in paths])
        actions = np.array([path["actions"] for path in paths])
        logits = np.array([path["logits"] for path in paths])
        rewards = np.array([path["rewards"] for path in paths])
        returns = np.array([path["returns"] for path in paths])
        values = np.array([path["values"] for path in paths])
        advantages = np.array([path["advantages"] for path in paths])
        has_feasible = all(p.get("feasible") is not None for p in paths)
        feasible = (
            np.array([p["feasible"] for p in paths]) if has_feasible else None
        )
        finish_time = np.array([path["finish_time"] for path in paths])
        
        # Handle energy if present (optional, for logging)
        if "energy" in paths[0] and paths[0]["energy"] is not None:
            energy = np.array([path["energy"] for path in paths])
        else:
            energy = None
        has_raw = all(p.get("raw_logits") is not None for p in paths)
        raw_logits = np.array([p["raw_logits"] for p in paths]) if has_raw else None
        return {
            "observations": observations,
            "actions": actions,
            "logits": logits,
            "rewards": rewards,
            "returns": returns,
            "values": values,
            "advantages": advantages,
            "finish_time": finish_time,
            "energy": energy,
            "feasible": feasible,
            "raw_logits": raw_logits,
        }

