import os
#import utils
import pickle
import os
import gc
import copy
import numpy as np
from utils import write_summary
import random

class MetaLearner(object):
    def __init__(self, sampler, policy, dic_agent_conf, dic_traffic_env_conf, dic_path):
        """
            Meta-learner incorporates MAML and MetaLight and can update the meta model by
            different learning methods.
            Arguments:
                sampler:    sample trajectories and update model parameters 
                policy:     frapplus_agent or metalight_agent
                ...
        """
        self.sampler = sampler
        self.policy = policy
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.meta_params = self.policy.save_params()
        self.meta_target_params = self.meta_params
        self.step_cnt = 0
        self.period = self.dic_agent_conf['PERIOD']

    def sample_maml(self, task, batch_id):
        """
            Use MAML framework to samples trajectories before and after the update of the parameters
            for all the tasks. Then, update meta-parameters.
        """
        self.batch_id = batch_id
        tasks = [task] * self.dic_traffic_env_conf['FAST_BATCH_SIZE']
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')
        learning_episodes = self.sampler.sample_maml(self.policy, tasks, batch_id, params=self.meta_params)
        self.policy.fit(learning_episodes, params=self.meta_params, target_params=self.meta_target_params)
        sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(learning_episodes))
        slice_index = random.sample(range(len(learning_episodes)), sample_size)
        params = self.policy.update_params(learning_episodes, params=copy.deepcopy(self.meta_params),
                                           lr_step=0, slice_index=slice_index)

        self.sampler.reset_task(tasks, batch_id, reset_type='meta')
        meta_episodes = self.sampler.sample_maml(self.policy, tasks, batch_id, params=params)
        self.policy.fit(meta_episodes, params=params, target_params=self.meta_target_params)

        sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(learning_episodes))
        slice_index = random.sample(range(len(learning_episodes)), sample_size)
        _grads = self.policy.cal_grads(learning_episodes, meta_episodes, slice_index=slice_index,
                                       params=self.meta_params)

        if self.dic_agent_conf['GRADIENT_CLIP']:
            for key in _grads.keys():
                _grads[key] = np.clip(_grads[key], -1 * self.dic_agent_conf['CLIP_SIZE'],
                                      self.dic_agent_conf['CLIP_SIZE'])
        with open(os.path.join(self.dic_path['PATH_TO_GRADIENT'], "gradients_%d.pkl")%batch_id,"ab+") as f:
            pickle.dump(_grads, f, -1)

        self.meta_params = params
        self.step_cnt += 1
        if self.step_cnt == self.period:
            self.step_cnt = 0
            self.meta_target_params = self.meta_params
        pickle.dump(self.meta_params, open(
            os.path.join(self.sampler.dic_path['PATH_TO_MODEL'],
                         'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))

    def sample_metalight(self, _tasks, batch_id):
        """
            Use MetaLight framework to samples trajectories before and after the update of the parameters
            for all the tasks. Then, update meta-parameters.
        """
        self.batch_id = batch_id
        tasks = []
        for task in _tasks:
            tasks.extend([task] * self.dic_traffic_env_conf[0]['FAST_BATCH_SIZE'])
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')
        meta_params = self.sampler.sample_metalight(self.policy, tasks, batch_id, params=self.meta_params,
                                       target_params=self.meta_target_params)
        pickle.dump(meta_params, open(
           os.path.join(self.sampler.dic_path[0]['PATH_TO_MODEL'], 'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))

    def sample_meta_test(self, task, batch_id, old_episodes=None):
        """
            Perform meta-testing (only testing within one episode) or offline-training (in multiple episodes to let models well trained and obtrained pre-trained models).
            Arguments:
                old_episodes: episodes generated and kept in former batches, controlled by 'MULTI_EPISODES'
                ...
        """
        self.batch_id = batch_id
        tasks = [task] * self.dic_traffic_env_conf['FAST_BATCH_SIZE']
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')

        self.meta_params, self.meta_target_params, episodes = \
            self.sampler.sample_meta_test(self.policy, tasks[0], batch_id, params=self.meta_params,
                                       target_params=self.meta_target_params, old_episodes=old_episodes)
        pickle.dump(self.meta_params, open(
            os.path.join(self.sampler.dic_path['PATH_TO_MODEL'], 'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))
        return episodes
