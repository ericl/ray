#!/usr/bin/env python
# Runs one or more regression tests. Retries tests up to 3 times.
#
# Example usage:
# $ python run_regression_tests.py regression-tests/cartpole-es-[tf|torch].yaml
#
# When using in BAZEL (with py_test), e.g. see in ray/rllib/BUILD:
# py_test(
#     name = "run_regression_tests",
#     main = "tests/run_regression_tests.py",
#     tags = ["learning_tests"],
#     size = "enormous",  # = 60min timeout
#     srcs = ["tests/run_regression_tests.py"],
#     data = glob(["tuned_examples/regression_tests/*.yaml"]),
#     Pass `BAZEL` option and the path to look for yaml regression files.
#     args = ["BAZEL", "tuned_examples/regression_tests"]
# )

from pathlib import Path
import sys
import yaml
import time
import os
import threading

import ray
from ray.tune import run_experiments
from ray.rllib import _register_all


# Manage timeouts internally. If we let the travis time limit be hit, we might
# never see the logs of the hanging test.
class Watchdog(threading.Thread):
    def __init__(self, timeout_seconds):
        threading.Thread.__init__(self)
        self.timeout_seconds = timeout_seconds
        self.daemon = True
        self.ok = False

    def reset(self):
        self.ok = True

    def run(self):
        while True:
            time.sleep(self.timeout_seconds)
            if not self.ok:
                print("Watchdog timed out after", self.timeout_seconds)
                os._exit(1)
            self.ok = False


if __name__ == "__main__":
    # Bazel regression test mode: Get path to look for yaml files from argv[2].
    if sys.argv[1] == "BAZEL":
        # Get the path to use.
        rllib_dir = Path(__file__).parent.parent
        print("rllib dir={}".format(rllib_dir))
        yaml_files = rllib_dir.rglob(sys.argv[2] + "/*.yaml")
        yaml_files = sorted(
            map(lambda path: str(path.absolute()), yaml_files), reverse=True)
    # Normal mode: Get yaml files to run from command line.
    else:
        yaml_files = sys.argv[1:]

    print("Will run the following regression files:")
    for yaml_file in yaml_files:
        print("->", yaml_file)

    watchdog = Watchdog(15 * 60)
    watchdog.start()

    # Loop through all collected files.
    for yaml_file in yaml_files:
        experiments = yaml.load(open(yaml_file).read())

        print("== Test config ==")
        print(yaml.dump(experiments))

        passed = False
        for i in range(3):
            try:
                watchdog.reset()
                ray.init(num_cpus=5)
                trials = run_experiments(experiments, resume=False, verbose=0)
            finally:
                ray.shutdown()
                _register_all()

            for t in trials:
                if (t.last_result["episode_reward_mean"] >=
                        t.stopping_criterion["episode_reward_mean"]):
                    passed = True
                    break

            if passed:
                print("Regression test PASSED")
                break
            else:
                print("Regression test FAILED on attempt {}", i + 1)

        if not passed:
            print("Overall regression FAILED: Exiting with Error.")
            sys.exit(1)
