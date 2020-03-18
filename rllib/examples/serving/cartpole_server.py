"""Example of running a policy server. Copy this file for your use case.

To try this out, in two separate shells run:
    $ python cartpole_server.py
    $ python cartpole_client.py [--local-inference]
"""

import argparse
import os

import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.tune.logger import pretty_print

SERVER_ADDRESS = "localhost"
SERVER_PORT = 9900
CHECKPOINT_FILE = "last_checkpoint.out"

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="DQN")

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    env = "CartPole-v0"
    connector_config = {
        # Use the connector server to generate experiences.
        "input": (
            lambda ioctx: PolicyServerInput( \
                ioctx, SERVER_ADDRESS, SERVER_PORT)
        ),
        # Use a single worker process to run the server.
        "num_workers": 0,
        # Disable OPE, since the rollouts are coming from online clients.
        "input_evaluation": [],
    }

    if args.run == "DQN":
        # Example of using DQN (supports off-policy actions).
        trainer = DQNTrainer(
            env=env,
            config=dict(
                connector_config, **{
                    "exploration_config": {
                        "type": "EpsilonGreedy",
                        "initial_epsilon": 1.0,
                        "final_epsilon": 0.02,
                        "epsilon_timesteps": 100,
                    },
                    "learning_starts": 100,
                    "timesteps_per_iteration": 200,
                    "log_level": "INFO",
                }))
    elif args.run == "PPO":
        # Example of using PPO (does NOT support off-policy actions).
        trainer = PPOTrainer(
            env=env,
            config=dict(
                connector_config, **{
                    "sample_batch_size": 1000,
                    "train_batch_size": 4000,
                }))
    else:
        raise ValueError("--run must be DQN or PPO")

    # Attempt to restore from checkpoint if possible.
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_path = open(CHECKPOINT_FILE).read()
        print("Restoring from checkpoint path", checkpoint_path)
        trainer.restore(checkpoint_path)

    # Serving and training loop
    while True:
        print(pretty_print(trainer.train()))
        checkpoint_path = trainer.save()
        print("Last checkpoint", checkpoint_path)
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(checkpoint_path)
