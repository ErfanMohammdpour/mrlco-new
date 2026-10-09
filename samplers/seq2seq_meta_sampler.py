from samplers.base import Sampler

from env.mec_offloaing_envs.scheduler import masking as _masking
from samplers.vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
from utils import utils, logger
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import os
import time
import itertools

# The TRAINING draw-space namespace. Imported from the CRN protocol module so there is exactly
# one definition of the phase constants: training identities always have parity
# `TRAINING_PHASE_SALT % 2 == 1` and evaluation identities (crn.seed_pair) parity 0, which is
# what makes the two draw spaces disjoint by construction (F18).
from spec.automotive_training.v2.crn import (
    PHASE_TRAIN, TRAINING_PHASE_SALT, draw_phase_of, training_seed_pair,
)

#: The v2 stateless, token-keyed sampling contract. `run_tf.py` stamps this on the trainer;
#: the sampler reports the branch it ACTUALLY took per rollout so a consumer can tell
#: `crn_token_keyed_v2` from the legacy stateful path instead of trusting a flag.
SAMPLING_CONTRACT_VERSION = "crn_token_keyed_v2"
DRAW_BRANCH_TOKEN_KEYED = "crn_token_keyed_v2"
DRAW_BRANCH_LEGACY_STATEFUL = "legacy_stateful_categorical"

class Seq2SeqMetaSampler(Sampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_policy_search.envs.base.MetaEnv) : environment object
        policy (meta_policy_search.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(self,
                env,
                policy,
                rollouts_per_meta_task,
                meta_batch_size,
                max_path_length,
                envs_per_task=None,
                parallel=False
                ):
        super(Seq2SeqMetaSampler, self).__init__(env, policy, rollouts_per_meta_task, max_path_length)
        assert hasattr(env, 'set_task')

        self.envs_per_task = rollouts_per_meta_task if envs_per_task is None else envs_per_task
        self.meta_batch_size = meta_batch_size
        self.total_samples = meta_batch_size * rollouts_per_meta_task * max_path_length
        self.parallel = parallel
        self.total_timesteps_sampled = 0

        # --- V2 DRAW IDENTITY (training phase) -------------------------------------------------
        # `draw_identity_seed` is the run seed set by the stack; the identity that reaches the
        # policy is `training_seed_pair(seed, cursor)`, i.e. an ODD first element, while every
        # evaluation identity from `crn.seed_pair` has an EVEN one. A training draw can
        # therefore never be replayed as, or confused with, an evaluation draw.
        self.draw_identity_seed = 0
        self._draw_cursor = 0
        #: phase namespace this sampler draws in ("train"); checked by `draw_identity()`.
        self.draw_phase = PHASE_TRAIN
        #: OBSERVED branch of the most recent rollout + how many times each branch ran. These
        #: are what `run_tf.py` records, so "the v2 sampler was used" is a measurement of the
        #: fetch that actually happened rather than a restatement of the request.
        self.draw_branch = None
        self.draw_branch_counts = {DRAW_BRANCH_TOKEN_KEYED: 0,
                                   DRAW_BRANCH_LEGACY_STATEFUL: 0}
        self.draw_key_prefix = None
        self.draw_cursor_path = "auto_sampler.sampler._draw_cursor"
        #: The REQUESTED contract. `run_tf.py --sampling-contract` records it on the trainer (see
        #: `set_sampling_contract`); the default is the v2 token-keyed contract. A non-v2 request
        #: deliberately does NOT pin an identity, so the draw really is the legacy stateful one
        #: instead of a v2 identity being requested and a legacy branch reported.
        self.sampling_contract_version = SAMPLING_CONTRACT_VERSION
        self.token_keyed_draws = True

        # setup vectorized environment
        if self.parallel:
            self.vec_env = MetaParallelEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)
        else:
            self.vec_env = MetaIterativeEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)

    def update_tasks(self):
        """
        Samples a new goal for each meta task
        """
        tasks = self.env.sample_tasks(self.meta_batch_size)
        assert len(tasks) == self.meta_batch_size
        self.vec_env.set_tasks(tasks)

        return tasks

    def set_sampling_contract(self, contract) -> str:
        """Record the REQUESTED sampling contract and resolve the draw branch.

        Called by `run_tf.py` with `--sampling-contract` (default `crn_token_keyed_v2`). A request
        for any other contract disables the identity pinning on purpose, so the run really uses
        the legacy stateful draw instead of reporting one branch and taking another.
        """
        self.sampling_contract_version = str(contract)
        self.token_keyed_draws = (str(contract) == SAMPLING_CONTRACT_VERSION)
        return self.sampling_contract_version

    def draw_identity(self, cursor=None) -> list:
        """The TRAINING draw identity `[seed, cursor]` handed to the policy for one rollout.

        Public (and pure) so the identity can be tested and journaled without running a
        rollout: `draw_identity(c)` is a function of the run seed and the cursor alone, and it
        is always in the `train` phase namespace.
        """
        c = int(self._draw_cursor if cursor is None else cursor)
        pair = training_seed_pair(int(getattr(self, "draw_identity_seed", 0) or 0), c)
        phase = draw_phase_of(pair)
        if phase != PHASE_TRAIN:
            raise RuntimeError("training draw identity %r is not in the %r phase"
                               % (pair, PHASE_TRAIN))
        return [int(pair[0]), int(pair[1])]

    def obtain_samples(self, log=False, log_prefix=''):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns:
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        paths = OrderedDict()
        for i in range(self.meta_batch_size):
            paths[i] = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy

        # initial reset of envs
        obses = self.vec_env.reset()

        while n_samples < self.total_samples:
            # execute policy
            t = time.time()
            # obs_per_task = np.split(np.asarray(obses), self.meta_batch_size)
            obs_per_task = np.array(obses)

            # V2 SAMPLING CONTRACT (F33/F20/F18): pin an IDENTITY-KEYED draw before this rollout.
            # Without it `get_actions` falls through to `Categorical(...).sample(seed=...)`, whose
            # op-level TF random stream is seeded once per process and whose ADVANCED DRAW HISTORY
            # is not part of a Saver checkpoint -- a resumed process then replays a different draw
            # sequence. The identity advances with `self._draw_cursor`, which is derived from the
            # sampler counters that ARE part of the resume bundle, so the sequence continues
            # exactly across a process boundary instead of restarting.
            #
            # The identity is built in the TRAINING phase namespace (`draw_identity`), so it can
            # never collide with an evaluation draw identity (F18).
            cursor = int(getattr(self, "_draw_cursor", 0))
            pin_pair = self.draw_identity(cursor)
            self._draw_cursor = cursor + 1
            pinned = False
            pin = getattr(policy, "set_draw_identity", None)
            if (pin is not None and getattr(policy, "enable_crn", False)
                    and getattr(self, "token_keyed_draws", True)):
                pin(pin_pair)
                pinned = True
            actions, logits, values = policy.get_actions(obs_per_task)
            # F20/F17: record the branch that ACTUALLY executed. A pinned identity is only
            # effective if the policy kept the stateless decoder reachable (`enable_crn` and the
            # crn_seed placeholder) and the decode did not fall back to the stateful sampler.
            observed = getattr(policy, "last_decode_branch", None)
            if observed is None:
                observed = (DRAW_BRANCH_TOKEN_KEYED if pinned
                            else DRAW_BRANCH_LEGACY_STATEFUL)
            if observed not in self.draw_branch_counts:
                self.draw_branch_counts[observed] = 0
            self.draw_branch_counts[observed] += 1
            self.draw_branch = observed
            self.draw_key_prefix = [int(pin_pair[0]), int(pin_pair[1])]
            # The mask the rollout actually used: store it, never recompute it at
            # update time, otherwise the PPO importance ratio stops being a
            # likelihood ratio (see reports/MASKED_PPO_INTERFACE_6b.md).
            applied_masks = getattr(policy, "last_feasible_masks", None)
            applied_raw = getattr(policy, "last_raw_logits", None)
            policy_time += time.time() - t

            # step environments
            t = time.time()
            # actions = np.concatenate(actions)

            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)

            # print("rewards shape is: ", np.array(rewards).shape)
            # print("finish time shape is: ", np.array(env_infos).shape)


            env_time += time.time() - t

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            new_samples = 0
            for idx, observation, action, logit, reward, value, done, env_info in zip(itertools.count(), obses, actions, logits,
                                                                                    rewards, values, dones, env_infos):
                # Handle energy in env_info (backward compatible). With scoped
                # telemetry enabled the env appends a third element; the legacy
                # 2-tuple path is untouched.
                if hasattr(self.env, 'resource_cluster') and self.env.resource_cluster.use_energy:
                    if isinstance(env_info, tuple) and len(env_info) >= 2:
                        task_finish_times, energy_info = env_info[0], env_info[1]
                        telemetry_info = env_info[2] if len(env_info) > 2 else None
                    else:
                        task_finish_times = env_info
                        energy_info = None
                        telemetry_info = None
                else:
                    task_finish_times = env_info
                    energy_info = None
                    telemetry_info = None
                
                # append new samples to running paths
                # handling
                # The policy's per-task arrays are already batched over this env's
                # graphs: dim0 is the SAME axis the loop below enumerates.
                # Pre-indexing by `idx % envs_per_task` stripped that axis, so each
                # path kept one token instead of the whole plan.
                task_mask = _masking.select_task_batch(applied_masks, idx, self.envs_per_task)
                if os.environ.get("MARGO_SHAPE_DEBUG") and idx == 0:
                    print("[shape-debug sampler] idx=0 observation=%s action=%s logit=%s mask=%s raw=%s" % (
                        np.shape(observation), np.shape(action), np.shape(logit),
                        None if applied_masks is None else np.shape(applied_masks[0]),
                        None if applied_raw is None else np.shape(applied_raw[0])), flush=True)
                task_raw = _masking.select_task_batch(applied_raw, idx, self.envs_per_task)

                for i, (single_ob, single_ac, single_logit, single_reward, single_value, single_task_finish_time) \
                        in enumerate(zip(observation, action, logit, reward, value, task_finish_times)):
                    running_paths[idx]["observations"]= single_ob
                    running_paths[idx]["actions"] = single_ac
                    running_paths[idx]["logits"] = single_logit
                    running_paths[idx]["rewards"] = single_reward
                    running_paths[idx]["finish_time"] = single_task_finish_time
                    running_paths[idx]["values"] = single_value
                    if task_mask is not None:
                        running_paths[idx]["feasible"] = task_mask[i]
                    if task_raw is not None:
                        running_paths[idx]["raw_logits"] = task_raw[i]
                    
                    # Store energy if enabled (energy_info[i] is the energy list for trajectory i)
                    if energy_info is not None and i < len(energy_info):
                        running_paths[idx]["energy"] = energy_info[i]

                    # Scoped telemetry travels with the same trajectory record.
                    if telemetry_info is not None and i < len(telemetry_info):
                        running_paths[idx]["energy_telemetry"] = telemetry_info[i]

                    path_dict = dict(
                        observations=np.squeeze(np.asarray(running_paths[idx]["observations"])),
                        actions=np.squeeze(np.asarray(running_paths[idx]["actions"])),
                        logits = np.squeeze(np.asarray(running_paths[idx]["logits"])),
                        rewards=np.squeeze(np.asarray(running_paths[idx]["rewards"])),
                        finish_time = np.squeeze(np.asarray(running_paths[idx]["finish_time"])),
                        values  = np.squeeze(np.asarray(running_paths[idx]["values"]))
                    )
                    if "feasible" in running_paths[idx]:
                        path_dict["feasible"] = np.squeeze(
                            np.asarray(running_paths[idx]["feasible"])
                        )
                    if "raw_logits" in running_paths[idx]:
                        path_dict["raw_logits"] = np.squeeze(
                            np.asarray(running_paths[idx]["raw_logits"])
                        )
                    
                    # Add energy to path if available
                    if "energy" in running_paths[idx]:
                        path_dict["energy"] = np.asarray(running_paths[idx]["energy"])
                    if running_paths[idx].get("energy_telemetry") is not None:
                        path_dict["energy_telemetry"] = running_paths[idx]["energy_telemetry"]
                    
                    paths[idx // self.envs_per_task].append(path_dict)

                # if running path is done, add it to paths and empty the running path
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)
        return paths

def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], logits=[], rewards=[])




