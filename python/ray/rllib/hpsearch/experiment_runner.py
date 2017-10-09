from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import ray
import sys

from ray.rllib.hpsearch.experiment import PENDING, RUNNING, TERMINATED
from ray.rllib.hpsearch.utils import gpu_count


class ExperimentRunner(object):

    def __init__(self, experiments):
        self._experiments = experiments
        self._pending = {}
        self._update_avail_resources()
        self._committed_resources = {k: 0 for k in self._avail_resources}

    def is_finished(self):
        for e in self._experiments:
            if e.status in [PENDING, RUNNING]:
                return False
        return True

    def can_launch_more(self):
        self._update_avail_resources()
        exp = self._get_runnable()
        return exp is not None

    def launch_experiment(self):
        exp = self._get_runnable()
        self._commit_resources(exp.resource_requirements())
        try:
            exp.start()
            self._pending[exp.train_remote()] = exp
        except:
            print("Error starting agent:", sys.exc_info()[0])
            self._return_resources(exp.resource_requirements())
            exp.stop(error=True)

    def process_events(self):
        [result_id], _ = ray.wait(self._pending.keys())
        exp = self._pending[result_id]
        del self._pending[result_id]
        try:
            result = ray.get(result_id)
            print("result", result)
            exp.update_progress(result)

            if exp.should_stop(result):
                self._return_resources(exp.resource_requirements())
                exp.stop()
            else:
                # TODO(rliaw): This implements checkpoint in a blocking manner
                if exp.should_checkpoint():
                    exp.checkpoint()
                self._pending[exp.train_remote()] = exp
        except:
            print("Error processing event:", sys.exc_info()[0])
            if exp.status == RUNNING:
                self._return_resources(exp.resource_requirements())
                exp.stop(error=True)

    def _get_runnable(self):
        for exp in self._experiments:
            if (exp.status == PENDING and
                    self._has_resources(exp.resource_requirements())):
                return exp
        return None

    def _has_resources(self, resources):
        for k, v in resources.items():
            if self._avail_resources[k] - self._committed_resources[k] < v:
                return False
        return True

    def _commit_resources(self, resources):
        for k, v in resources.items():
            self._committed_resources[k] += v
            assert self._avail_resources[k] >= 0

    def _return_resources(self, resources):
        for k, v in resources.items():
            self._committed_resources[k] -= v
            assert self._committed_resources[k] >= 0

    def _update_avail_resources(self):
        clients = ray.global_state.client_table()
        local_schedulers = [    
            entry for client in clients.values() for entry in client
                if entry['ClientType'] == 'local_scheduler' \
                    and not entry['Deleted']]
        num_clients = len(local_schedulers)
        num_cpus = sum(ls['NumCPUs'] for ls in local_schedulers)
        num_gpus = sum(ls['NumGPUs'] for ls in local_schedulers)
        self._avail_resources = {
            'cpu': int(num_cpus),
            'gpu': int(num_gpus),
        }

    def debug_string(self):
        statuses = [
            ' - {}:\t{}'.format(e, e.progress_string())
                for e in self._experiments]
        return 'Available resources: {}'.format(self._avail_resources) + \
            '\nCommitted resources: {}'.format(self._committed_resources) + \
            '\nAll experiments:\n' + '\n'.join(statuses)

