from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import random

from ray.tune.logger import pretty_print
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.trial import Trial


class CheckpointBasedPruning(FIFOScheduler):
    """Implements checkpoint-based pruning.

    This scheduler will prune hyperparameter combinations based on an estimate
    or their long-term performance. This estimate is computed by training an
    existing checkpoint with the given combination a few iterations.

    Args:
        reltime_attr (str): A result attribute that measures time
            since the restore only (e.g., iterations_since_restore).
        reward_attr (str): The training result objective value attribute. As
            with `time_attr`, this may refer to any objective value. Stopping
            procedures will use this attribute.
        bootstrap_checkpoint (str): Existing checkpoint to use for pruning.
            If not specified, an initial one will be generated from trial runs.
        checkpoint_min_reward (float): Start taking checkpoints for pruning
            after this reward is reached. Note: set this to infinity if you
            only want to use the bootstrap checkpoint.
        checkpoint_eval_t (float): Amount of additional time to evaluate
            hyperparams starting from the checkpoint time.
        reduction_factor (float): Controls aggressiveness of pruning. Must be
            set to a value f > 1.
        brackets (int): Number of brackets. Each bracket has a different
            halving rate, specified by the reduction factor.
    """

    def __init__(self,
                 reltime_attr="iterations_since_restore",
                 reward_attr="episode_reward_mean",
                 bootstrap_checkpoint=None,
                 checkpoint_min_reward=0,
                 checkpoint_eval_t=1,
                 reduction_factor=5):
        assert reduction_factor > 1, "Reduction Factor not valid!"
        FIFOScheduler.__init__(self)
        self._reltime_attr = reltime_attr
        self._reward_attr = reward_attr
        self._current_reward = checkpoint_min_reward
        self._current_checkpoint = bootstrap_checkpoint
        self._checkpoint_min_reward = checkpoint_min_reward
        self._checkpoint_eval_t = checkpoint_eval_t
        self._reduction_factor = reduction_factor

        # map of eval trials to the original requested trial
        self._eval_trials = {}
        self._waiting_for_eval = set()
        self._eval_time = 0.0

        # set of trials we are running for real
        self._admitted_trials = set()
        self._run_time = 0.0

        # map from score to original requested trial
        self._eval_scores = []
        self._num_eval = 0
        self._num_run = 0

    def choose_trial_to_run(self, trial_runner):
        # Don't have a checkpoint yet, so behave like the default scheduler
        if not self._current_checkpoint:
            return self._choose_random_trial(trial_runner)

        # Have some admissable trials ready to run
        admissable = self._get_admissable_trials()
        random.shuffle(admissable)
        if admissable:
            for trial in admissable:
                if trial_runner.has_resources(trial.resources):
                    self._admitted_trials.add(trial)
                    self._num_run += 1
                    return trial

        # Launch filtering tasks based on the checkpoint
        trials = list(trial_runner.get_trials())
        random.shuffle(trials)
        for trial in trials:
            if (trial.status == Trial.PENDING and
                    trial not in self._waiting_for_eval and
                    trial_runner.has_resources(trial.resources)):
                print("Launch eval", trial.config)
                eval_trial = Trial(
                    trial.trainable_name,
                    config=trial.config,
                    local_dir=os.path.join(
                        os.path.dirname(trial.local_dir),
                        os.path.basename(trial.local_dir) + "_chkpt_eval"),
                    experiment_tag="0_chkpt_eval_{}".format(self._num_eval),
                    resources=trial.resources,
                    stopping_criterion={
                        self._reltime_attr: self._checkpoint_eval_t,
                    },
                    restore_path=self._current_checkpoint,
                    upload_dir=trial.upload_dir)
                trial_runner.add_trial(eval_trial)
                self._eval_trials[eval_trial] = trial
                self._waiting_for_eval.add(trial)
                self._num_eval += 1
                return eval_trial

        # Nothing to do, fall back to running remaining trials
        return self._choose_random_trial(trial_runner)

    def _choose_random_trial(self, trial_runner):
        trials = list(trial_runner.get_trials())
        random.shuffle(trials)
        for trial in trials:
            if (trial.status == Trial.PENDING
                    and trial_runner.has_resources(trial.resources)):
                self._num_run += 1
                return trial
        for trial in trials:
            if (trial.status == Trial.PAUSED
                    and trial_runner.has_resources(trial.resources)):
                self._num_run += 1
                return trial
        return None

    def on_trial_result(self, trial_runner, trial, result):
        score = result[self._reward_attr]
        if score > self._current_reward * 1.1:
            print("Resetting checkpoint due to new high score", score)
            self._current_checkpoint = trial_runner.trial_executor.save(trial)
            self._current_reward = score
            self._waiting_for_eval.clear()
            self._eval_trials.clear()
            self._eval_scores.clear()
        return TrialScheduler.CONTINUE

    def on_trial_complete(self, trial_runner, trial, result):
        if trial in self._eval_trials:
            self._eval_time += result[self._reltime_attr]
            orig_trial = self._eval_trials[trial]
            del self._eval_trials[trial]
            self._record_eval_result(result, orig_trial)
        elif trial in self._admitted_trials:
            self._run_time += result[self._reltime_attr]
        else:
            print("WARN: Ignoring stale eval result from", trial, result)

    def _record_eval_result(self, eval_result, orig_trial):
        score = eval_result[self._reward_attr]
        if not np.isnan(score):
            self._eval_scores.append((score, orig_trial))

    def _get_admissable_trials(self):
        if len(self._eval_scores) < self._reduction_factor:
            return []
        threshold = np.percentile(
            [t[0] for t in self._eval_scores],
            q=100 * (1 - 1.0 / self._reduction_factor))
        res = []
        for score, orig_trial in self._eval_scores:
            if score > threshold and orig_trial not in self._admitted_trials:
                res.append(orig_trial)
        return res

    def debug_string(self):
        scores = [t[0] for t in self._eval_scores]
        info = {
            "eval_time": self._eval_time,
            "eval_count": self._num_eval,
            "eval_scores_mean": np.mean(scores),
            "eval_scores_max": np.max(scores) if scores else float("nan"),
            "eval_scores_min": np.min(scores) if scores else float("nan"),
            "run_time": self._run_time,
            "run_count": self._num_run,
        }
        info = "  {}".format(pretty_print(info).replace("\n", "\n  "))
        return "CheckpointBasedPruning:\n{}".format(info).strip()
