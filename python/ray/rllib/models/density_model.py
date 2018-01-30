import sys

import numpy as np
from ray.rllib.bc.experience_dataset import ExperienceDataset
from ray.rllib.models.fast_cts import CTSDensityModel


class DensityModel(object):
    def __init__(self, num_actions=None):
        self.model_actions = num_actions is not None
        if self.model_actions:
            self.act_cnt = np.zeros(num_actions)
            self.models = [CTSDensityModel() for _ in range(num_actions)]
        else:
            self.model = CTSDensityModel()

    def train(self, dataset, num_samples, idx_range=None):
        for _ in range(num_samples):
            if _ % 1000 == 0:
                print("Training density model", _)
            sample = dataset.sample(1, idx_range)
            o = sample['observations'][0]
            a = sample['actions'][0]
            if self.model_actions:
                self.act_cnt[a] += 1
                self.models[a].update(o[:, :, 0])
            else:
                self.model.update(o[:, :, 0])

    def logp(self, obs, action):
        if self.model_actions:
            log_act_cnt = np.log(self.act_cnt)
            log_cnt = np.log(self.act_cnt.sum())
            self.models[action].log_prob(obs[:, :, 0])
            return log_act_cnt[action] - log_cnt + self.models[action].log_prob(obs[:, :, 0])
        else:
            return self.model.log_prob(obs[:, :, 0])

    def test(self, dataset, num_samples, idx_range=None):
        sample = dataset.sample(num_samples, idx_range)
        if self.model_actions:
            log_act_cnt = np.log(self.act_cnt)
            log_cnt = np.log(self.act_cnt.sum())
            return (sum(log_act_cnt[a] for a in sample['actions']) - log_cnt * num_samples
                    + sum(self.models[sample['actions'][i]].log_prob(sample['observations'][i][:, :, 0]) for i in range(num_samples)))
        else:
            return sum(self.model.log_prob(o[:, :, 0]) for o in sample['observations'])


def main():
    dataset = ExperienceDataset({sys.argv[1]: .1})
    models = []
    for idx_range in [(0., .1), (.9, 1.)]:
        model = DensityModel(6)
        model.train(dataset, 1000, idx_range)
        models.append(model)
    models[0].logp(dataset.sample(1)['observations'][0], 0)
    scores = [[model.test(dataset, 1000, idx_range) for idx_range in [(i / 10., (i + 1) / 10.) for i in range(10)]] for model in models]
    print(scores)


if __name__ == "__main__":
    main()
