from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import os
import random
import pickle

import cv2
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
                for i in indexes:
                    obs.append(self.make_obs(f["rgb"][i], f["targets"][i]))
                    actions.append(self.make_action(f["targets"][i]))
        samples = {
            'observations': obs,
            'actions': actions,
        }
        return samples

    # https://github.com/carla-simulator/imitation-learning
    def make_obs(self, rgb, target):
        data = rgb.reshape(200, 88, 3)  # original size
        data = cv2.resize(
            data, (80, 80),
            interpolation=cv2.INTER_AREA)  # resize for network input
        data = (data.astype(np.float32) - 128) / 128

        control_signal = target[24]
        forward_speed = target[10]
        dist_to_goal = 0  # not provided?

        if control_signal == 2:
            command = [0, 0, 0, 0, 1]
        elif control_signal == 3:
            command = [0, 0, 0, 1, 0]
        elif control_signal == 4:
            command = [0, 0, 1, 0, 0]
        elif control_signal == 5:
            command = [0, 1, 0, 0, 0]
        else:
            command = [1, 0, 0, 0, 0]

        return (data, command, [forward_speed, dist_to_goal])

    def make_action(self, target):
        steer = target[0]
        gas = target[1]
        brake = target[2]
        return gas if gas >= 0 else -brake, steer
