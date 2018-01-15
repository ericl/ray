from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import os
import random
import pickle

import h5py
import numpy as np


class ExperienceDataset(object):
    def __init__(self, dataset_path):
        self._dataset = list(itertools.chain.from_iterable(pickle.load(open(dataset_path, "rb"))))

    def sample(self, batch_size):
        indexes = np.random.choice(len(self._dataset), batch_size)
        samples = {
            'observations': [self._dataset[i][0] for i in indexes],
            'actions': [self._dataset[i][1] for i in indexes]
        }
        return samples


class HDF5Dataset(object):
    def __init__(self, dataset_path):
        self._files = glob.glob(os.path.expanduser(dataset_path) + "/*.h5")
        assert len(self._files) > 0

    def sample(self, batch_size):
        def random_file():
            return h5py.File(random.choice(self._files))
        obs = []
        actions = []
        while len(obs) < batch_size:
            with random_file() as f:
                indexes = np.random.choice(len(f["targets"]), batch_size)
                obs.append([f["rgb"][i] for i in indexes])
                actions.append([f["targets"][i] for i in indexes])
        samples = {
            'observations': obs,
            'actions': actions,
        }
        print("Returning", samples)
        return samples
