from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.agents.trainer import Trainer
from ray.rllib.utils.annotations import override, DeveloperAPI


@DeveloperAPI
def build_trainer(name,
                  default_config,
                  default_policy_graph,
                  make_policy_optimizer,
                  validate_config=None,
                  get_policy_graph=None,
                  before_train_step=None,
                  after_optimizer_step=None,
                  after_train_result=None):
    """Helper function for defining a custom trainer.

    Arguments:
        name (str): name of the trainer (e.g., "PPO")
        default_config (dict): the default config dict of the algorithm
        default_policy_graph (cls): the default PolicyGraph class to use
        make_policy_optimizer (func): function that returns a PolicyOptimizer
            instance given (local_evaluator, remote_evaluators, config)
        validate_config (func): optional callback that checks a given config
            for correctness. It may mutate the config as needed.
        get_policy_graph (func): optional callback that takes a config and
            returns the policy graph class to override the default with
        before_train_step (func): optional callback to run before each train()
            call. It takes the trainer instance as an argument.
        after_optimizer_step (func): optional callback to run after each
            step() call to the policy optimizer. It takes the trainer instance
            and the policy gradient fetches as arguments.
        after_train_result (func): optional callback to run at the end of each
            train() call. It takes the trainer instance and result dict as
            arguments, and may mutate the result dict as needed.

    Returns:
        a Trainer instance that uses the specified args.
    """

    class trainer_cls(Trainer):
        _name = name
        _default_config = default_config
        _policy_graph = default_policy_graph

        def _init(self, config, env_creator):
            if validate_config:
                validate_config(config)
            if get_policy_graph is None:
                policy_graph = default_policy_graph
            else:
                policy_graph = get_policy_graph(config)
            self.local_evaluator = self.make_local_evaluator(
                env_creator, policy_graph)
            self.remote_evaluators = self.make_remote_evaluators(
                env_creator, policy_graph, config["num_workers"])
            if make_policy_optimizer:
                self.optimizer = make_policy_optimizer(
                    self.local_evaluator, self.remote_evaluators, config)

        @override(Trainer)
        def _train(self):
            if before_train_step:
                before_train_step(self)
            prev_steps = self.optimizer.num_steps_sampled
            fetches = self.optimizer.step()
            if after_optimizer_step:
                after_optimizer_step(self, fetches)
            res = self.collect_metrics()
            res.update(
                timesteps_this_iter=self.optimizer.num_steps_sampled -
                prev_steps,
                info=res.get("info", {}))
            if after_train_result:
                after_train_result(self, res)
            return res

    trainer_cls.__name__ = name
    return trainer_cls
