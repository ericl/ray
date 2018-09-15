from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler


class CheckpointBasedPruning(FIFOScheduler):
    """Implements checkpoint-based pruning.

    This scheduler will prune hyperparameter combinations based on an estimate
    or their long-term performance. This estimate is computed by training an
    existing checkpoint with the given combination a few iterations.

    Args:
        time_attr (str): A training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        reltime_attr (str): The corresponding attr that measures time
            since the restore only (e.g., iterations_since_restore).
        reward_attr (str): The training result objective value attribute. As
            with `time_attr`, this may refer to any objective value. Stopping
            procedures will use this attribute.
        bootstrap_checkpoint (str): Existing checkpoint to use for pruning.
            If not specified, an initial one will be generated from trial runs.
        checkpoint_min_reward (float): Start taking checkpoints for pruning
            after this reward is reached.
        checkpoint_eval_t (float): Amount of additional time to evaluate
            hyperparams starting from the checkpoint time.
        reduction_factor (float): Controls aggressiveness of pruning. Must be
            set to a value f > 1.
        brackets (int): Number of brackets. Each bracket has a different
            halving rate, specified by the reduction factor.
    """

    def __init__(self,
                 time_attr="training_iteration",
                 reltime_attr="iterations_since_restore",
                 reward_attr="episode_reward_mean",
                 bootstrap_checkpoint=None,
                 checkpoint_min_reward=50,
                 checkpoint_eval_t=1,
                 reduction_factor=5):
        assert bootstrap_checkpoint, "TODO: auto generate these"
        assert reduction_factor > 1, "Reduction Factor not valid!"
        FIFOScheduler.__init__(self)
        self._time_attr = time_attr
        self._reltime_attr = reltime_attr
        self._reward_attr = reward_attr
        self._current_reward = checkpoint_min_reward
        self._current_checkpoint = bootstrap_checkpoint
        self._checkpoint_min_reward = checkpoint_min_reward
        self._checkpoint_eval_t = checkpoint_eval_t
        self._reduction_factor = reduction_factor

        # map of eval trials to the original requested trial
        self._eval_trials = {}
        self._eval_time = 0.0

        # set of trials we are running for real
        self._admitted_trials = set()
        self._run_time = 0.0

        # map from score to original requested trial
        self._eval_scores = {}
        self._num_eval = 0
        self._num_run = 0

    def choose_trial_to_run(self, trial_runner):
        # Don't have a checkpoint yet, so behave like the default scheduler
        if not self._current_checkpoint:
            self._num_run += 1
            return FIFOScheduler.choose_trial_to_run(self)

        # Have some admissable trials ready to run
        admissable = self._get_admissable_trials()
        if admissable:
            for trial in admissable:
                if trial_runner.has_resources(trial.resources):
                    self._admitted_trials.add(trial)
                    self._num_run += 1
                    return trial
            return None

        # Launch filtering tasks based on the checkpoint
        for trial in trial_runner.get_trials():
            if (trial.status = Trial.PENDING and
                    trial_runner.has_resources(trial.resources)):
                eval_trial = Trial(
                    trial.trainable_name,
                    config=trial.config,
                    local_dir=trial.local_dir,
                    experiment_tag="cbp_eval_{}".format(trial.experiment_tag),
                    resources=trial.resources,
                    stopping_criterion={
                        self._reltime_attr: self._checkpoint_eval_t,
                    },
                    restore_path=self._current_checkpoint,
                    upload_dir=trial.upload_dir)
                trial_runner.add_trial(eval_trial)
                self._eval_trials[eval_trial] = trial
                self._num_eval += 1
                return eval_trial
        return None

    def on_trial_complete(self, trial_runner, trial, result):
        if trial in self._eval_trials:
            self._eval_time += result["time_since_restore"]
            orig_trial = self._eval_trials[trial]
            del self._eval_trials[trial]
            self._record_eval_result(result, orig_trial)

    def _record_eval_result(self, eval_result, orig_trial):
        score = eval_result[self._reward_attr]
        self._eval_scores[score] = orig_trial

    def _admissable_trials(self):
        if len(self._eval_scores) < 4:
            return []
        threshold = np.percentile(
            list(self._eval_scores.keys()),
            q=100 * (1 - 1.0 / self._reduction_factor))
        res = []
        for score, orig_trial in self._eval_scores():
            if score > threshold and orig_trial not in self._admitted_trials:
                res.append(orig_trial)
        return res

    def debug_string(self):
        info = {
            "eval_time": self._eval_time,
            "eval_count": self._num_eval,
            "eval_scores": sorted(list(self._eval_scores.keys())),
            "run_time": self._run_time,
            "run_count": self._num_run,
        }
        info = "  {}".format(pretty_print(info).replace("\n", "\n  "))
        return "CheckpointBasedPruning: {}".format(info)
