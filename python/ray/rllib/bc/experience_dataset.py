from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import json
import itertools
import pickle
import io

import base64
import numpy as np


def unpack(obj):
    if "data" not in obj:
        return obj
    data = np.load(io.BytesIO(base64.b64decode(obj["data"])))
    del obj["data"]
    for k, v in data.items():
        obj[k] = v
    return obj


class ExperienceDataset(object):
    def __init__(self, dataset_path):
        """Create dataset of experience to imitate.

        Parameters
        ----------
        dataset_path:
          Path of file containing the database as pickled list of trajectories,
          each trajectory being a list of steps,
          each step containing the observation and action as its first two elements.
          The file must be available on each machine used by a BCEvaluator.
        """
#        self._dataset = list(itertools.chain.from_iterable(pickle.load(open(dataset_path, "rb"))))
        self._dataset = []
        for path, sample_frac in dataset_path.items():
            for line in open(path).read().split("\n"):
                line = line.strip()
                if line and random.random() < sample_frac:
                    self._dataset.append(unpack(json.loads(line)))
            print("Loaded dataset size", len(self._dataset))

    def sample(self, batch_size, idx_range=None):
        if idx_range is None:
            idx_range = (0., 1.)
        idx_range = tuple(int(i * len(self._dataset)) for i in idx_range)
        indexes = np.random.choice(idx_range[1] - idx_range[0], batch_size) + idx_range[0]
        samples = {
            'observations': [self._dataset[i]["obs"] for i in indexes],
            'actions': [self._dataset[i]["action"] for i in indexes],
        }
        return samples
