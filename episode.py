import numpy as np
import copy
import  config

class BatchEpisodes(object):
    def __init__(self, dic_agent_conf, old_episodes=None):
        self.dic_agent_conf = dic_agent_conf

        self.total_samples = []

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None
        self.tot_x = []
        self.tot_next_x = []
        if old_episodes:
            self.total_samples = self.total_samples + old_episodes.total_samples
            self.tot_x = self.tot_x + old_episodes.tot_x
            self.tot_next_x = self.tot_next_x + old_episodes.tot_next_x

        self.last_x = []
        self.last_next_x = []
        self.current_x = []
        self.current_next_x = []

    def append(self, observations, actions, new_observations, rewards, batch_ids):
        self.last_x = self.current_x
        self.last_next_x = self.current_next_x
        self.current_x = []
        self.current_next_x = []

        for observation, action, new_observation, reward, batch_id in zip(
                observations, actions, new_observations, rewards, batch_ids):
            if batch_id is None:
                continue

            self.total_samples.append([observation, action, new_observation, reward, 0, 0])

            self.tot_x.append(observation[0]['lane_num_vehicle'] + observation[0]["cur_phase"])
            self.current_x.append(observation[0]['lane_num_vehicle'] + observation[0]["cur_phase"])

            self.tot_next_x.append(new_observation[0]['lane_num_vehicle'] + new_observation[0]["cur_phase"])
            self.current_next_x.append(new_observation[0]['lane_num_vehicle'] + new_observation[0]["cur_phase"])

    def get_x(self):
        return np.reshape(np.array(self.tot_x), (len(self.tot_x), -1))

    def get_next_x(self):
        return np.reshape(np.array(self.tot_next_x), (len(self.tot_next_x), -1))

    def forget(self):
        self.total_samples = self.total_samples[-1 * self.dic_agent_conf['MAX_MEMORY_LEN'] : ]
        self.tot_x = self.tot_x[-1 * self.dic_agent_conf['MAX_MEMORY_LEN'] : ]
        self.tot_next_x = self.tot_next_x[-1 * self.dic_agent_conf['MAX_MEMORY_LEN']:]

    def prepare_y(self, q_values):
        self.tot_y = q_values

    def get_y(self):
        return self.tot_y

    def __len__(self):
        return len(self.total_samples)

class SeperateEpisode:
    def __init__(self, size, group_size, dic_agent_conf, old_episodes=None):
        self.episodes_inter = []
        for _ in range(size):
            self.episodes_inter.append(BatchEpisodes(
                dic_agent_conf=dic_agent_conf, old_episodes=old_episodes)
            )
        self.num_group = size
        self.group_size = group_size

    def append(self, observations, actions, new_observations, rewards, batch_ids):

        for i in range(int(len(observations) / self.group_size)):
            a = i * self.group_size
            b = (i + 1) * self.group_size
            self.episodes_inter[i].append(observations[a : b], actions[a : b],
                                          new_observations[a : b], rewards[a : b], batch_ids)
        #for i in range(len(self.episodes_inter)):
        #    self.episodes_inter[i].append(observations[:, i], actions[:, i], new_observations[:, i], rewards[:, i], batch_ids)

    def forget(self, memory_len):
        for i in range(len(self.episodes_inter)):
            self.episodes_inter[i].forget(memory_len)

    def __len__(self):
        return len(self.episodes_inter[0].total_samples)
