import multiprocessing as mp
from episode import BatchEpisodes, SeperateEpisode
from cityflow_env import CityFlowEnv
import json
import os
import shutil
import random
import copy
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from math import isnan
from subproc_vec_env import SubprocVecEnv
from utils import write_summary
import pickle

class BatchSampler(object):
    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                           dic_path, batch_size, num_workers=2):
        """
            Sample trajectories in one episode by different methods
        """
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.task_path_map = {}
        self.task_traffic_env_map = {}
        if not isinstance(self.dic_traffic_env_conf, list):
            self.list_traffic_env_conf = [self.dic_traffic_env_conf]
            self.list_path = [self.dic_path]
            task = self.dic_path["PATH_TO_DATA"].split("/")[-1] + ".json"
            self.task_path_map[task] = self.dic_path
            self.task_traffic_env_map[task] = self.dic_traffic_env_conf
        else:
            self.list_traffic_env_conf = self.dic_traffic_env_conf
            self.list_path = self.dic_path
            for path in self.dic_path:
                task = path["PATH_TO_DATA"].split("/")[-1] + ".json"
                self.task_path_map[task] = path
            for env in self.dic_traffic_env_conf:
                task = env["TRAFFIC_FILE"]
                self.task_traffic_env_map[task] = env

        # num of episodes
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.queue = mp.Queue()
        self.envs = None
        self._task_id = 0

        self._path_check()
        self._copy_conf_file()
        # self._copy_cityflow_file() 

        self.path_to_log = self.list_path[0]['PATH_TO_WORK_DIRECTORY']

        self.step = 0
        self.target_step = 0
        self.lr_step = 0

        self.test_step = 0

    def _path_check(self):
        # check path
        if not os.path.exists(self.list_path[0]["PATH_TO_WORK_DIRECTORY"]):
            os.makedirs(self.list_path[0]["PATH_TO_WORK_DIRECTORY"])

        if not os.path.exists(self.list_path[0]["PATH_TO_MODEL"]):
            os.makedirs(self.list_path[0]["PATH_TO_MODEL"])

        if not os.path.exists(self.list_path[0]["PATH_TO_GRADIENT"]):
            os.makedirs(self.list_path[0]["PATH_TO_GRADIENT"])

        if self.dic_exp_conf["PRETRAIN"]:
            if os.path.exists(self.list_path[0]["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
                pass
            else:
                os.makedirs(self.list_path[0]["PATH_TO_PRETRAIN_WORK_DIRECTORY"])

            if os.path.exists(self.list_path[0]["PATH_TO_PRETRAIN_MODEL"]):
                pass
            else:
                os.makedirs(self.list_path[0]["PATH_TO_PRETRAIN_MODEL"])

    def _copy_conf_file(self, path=None):
        # write conf files
        if path == None:
            path = self.list_path[0]["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
                  indent=4)
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    def _copy_cityflow_file(self, path=None):
        if path == None:
            path = self.list_path[0]["PATH_TO_WORK_DIRECTORY"]

        for traffic in self.dic_exp_conf["TRAFFIC_IN_TASKS"]:
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], traffic),
                            os.path.join(path, traffic))
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_traffic_env_conf["ROADNET_FILE"]),
                        os.path.join(path, self.dic_traffic_env_conf["ROADNET_FILE"]))

    def sample_maml(self, policy, task=None, batch_id=None, params=None):
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        episodes = BatchEpisodes(dic_agent_conf=self.dic_agent_conf)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        if params: # todo precise load parameter logic
            policy.load_params(params)
        while (not all(dones)) or (not self.queue.empty()):
            actions = policy.choose_action(observations)
            ## for multi_intersection
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, new_observations, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        #self.envs.bulk_log()
        return episodes

    def sample_sotl(self, policy, task=None, batch_id=None, params=None):
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        if params: # todo precise load parameter logic
            policy.load_params(params)
        while (not all(dones)):
            actions = policy.choose_action(observations)
            ## for multi_intersection
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            observations, batch_ids = new_observations, new_batch_ids
        write_summary(self.dic_path, task, self.dic_exp_conf["EPISODE_LEN"], 0,
                      self.dic_traffic_env_conf['FLOW_FILE'])
        #self.envs.bulk_log()

    def sample_metalight(self, policy, tasks, batch_id, params=None, target_params=None, episodes=None):
        for i in range(len(tasks)):
            self.queue.put(i)
        for _ in range(len(tasks)):
            self.queue.put(None)

        if not episodes:
            size = int(len(tasks) / self.list_traffic_env_conf[0]["FAST_BATCH_SIZE"])
            episodes = SeperateEpisode(size=size, group_size=self.list_traffic_env_conf[0]["FAST_BATCH_SIZE"],
                                       dic_agent_conf=self.dic_agent_conf)

        observations, batch_ids = self.envs.reset()
        dones = [False]
        if params: # todo precise load parameter logic
            policy.load_params(params)

        old_params = None
        meta_update_period = 1
        meta_update = False

        while (not all(dones)) or (not self.queue.empty()):
            actions = policy.choose_action(observations)
            ## for multi_intersection
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, new_observations, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids

            # if update
            if self.step > self.dic_agent_conf['UPDATE_START'] and self.step % self.dic_agent_conf['UPDATE_PERIOD'] == 0:

                if len(episodes) > self.dic_agent_conf['MAX_MEMORY_LEN']:
                    #TODO
                    episodes.forget()

                old_params = params

                policy.fit(episodes, params=params, target_params=target_params)
                sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(episodes))
                slice_index = random.sample(range(len(episodes)), sample_size)
                params = policy.update_params(episodes, params=copy.deepcopy(params),
                                              lr_step=self.lr_step, slice_index=slice_index)
                policy.load_params(params)

                self.target_step += 1
                if self.target_step == self.dic_agent_conf['UPDATE_Q_BAR_FREQ']:
                    target_params = params
                    self.target_step = 0

                # meta update
                if meta_update_period % self.dic_agent_conf["META_UPDATE_PERIOD"] == 0:
                    policy.fit(episodes, params=params, target_params=target_params)
                    sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(episodes))
                    new_slice_index = random.sample(range(len(episodes)), sample_size)
                    params = policy.update_meta_params(episodes, slice_index, new_slice_index, _params=old_params)
                    policy.load_params(params)

                meta_update_period += 1

            self.step += 1

        if not meta_update:
            policy.fit(episodes, params=params, target_params=target_params)
            sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(episodes))
            new_slice_index = random.sample(range(len(episodes)), sample_size)
            params = policy.update_meta_params(episodes, slice_index, new_slice_index, _params=old_params)
            policy.load_params(params)

            meta_update_period += 1
        policy.decay_epsilon(batch_id)
        return params[0]

        #self.envs.bulk_log()

    def sample_meta_test(self, policy, task, batch_id, params=None, target_params=None, old_episodes=None):
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        episodes = BatchEpisodes(dic_agent_conf=self.dic_agent_conf, old_episodes=old_episodes)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        if params: # todo precise load parameter logic
            policy.load_params(params)

        while (not all(dones)) or (not self.queue.empty()):
            actions = policy.choose_action(observations)
            ## for multi_intersection
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, new_observations, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids

            if self.step > self.dic_agent_conf['UPDATE_START'] and self.step % self.dic_agent_conf['UPDATE_PERIOD'] == 0:
                if len(episodes) > self.dic_agent_conf['MAX_MEMORY_LEN']:
                    episodes.forget()

                policy.fit(episodes, params=params, target_params=target_params)
                sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(episodes))
                slice_index = random.sample(range(len(episodes)), sample_size)
                params = policy.update_params(episodes, params=copy.deepcopy(params),
                                              lr_step=self.lr_step, slice_index=slice_index)

                policy.load_params(params)

                self.lr_step += 1
                self.target_step += 1
                if self.target_step == self.dic_agent_conf['UPDATE_Q_BAR_FREQ']:
                    target_params = params
                    self.target_step = 0

            if self.step > self.dic_agent_conf['UPDATE_START'] and self.step % self.dic_agent_conf['TEST_PERIOD'] == 0:
                self.single_test_sample(policy, task, self.test_step, params=params)
                pickle.dump(params, open(
                    os.path.join(self.dic_path['PATH_TO_MODEL'], 'params' + "_" + str(self.test_step) + ".pkl"),
                    'wb'))
                write_summary(self.dic_path, task,
                              self.dic_traffic_env_conf["EPISODE_LEN"], batch_id)

                self.test_step += 1
            self.step += 1

        policy.decay_epsilon(batch_id)
        self.envs.bulk_log()
        return params, target_params, episodes

    def single_test_sample(self, policy, task, batch_id, params):
        policy.load_params(params)

        dic_traffic_env_conf = copy.deepcopy(self.dic_traffic_env_conf)
        dic_traffic_env_conf['TRAFFIC_FILE'] = task

        dic_path = copy.deepcopy(self.dic_path)
        dic_path["PATH_TO_LOG"] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], 'test_round',
                                              task, 'tasks_round_' + str(batch_id))

        if not os.path.exists(dic_path['PATH_TO_LOG']):
            os.makedirs(dic_path['PATH_TO_LOG'])

        dic_exp_conf = copy.deepcopy(self.dic_exp_conf)

        env = CityFlowEnv(path_to_log=dic_path["PATH_TO_LOG"],
                      path_to_work_directory=dic_path["PATH_TO_DATA"],
                      dic_traffic_env_conf=dic_traffic_env_conf)

        done = False
        state = env.reset()
        step_num = 0
        stop_cnt = 0
        while not done and step_num < int(
                dic_exp_conf["EPISODE_LEN"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            for one_state in state:
                action = policy.choose_action([[one_state]], test=True) # one for multi-state, the other for multi-intersection
                action_list.append(action[0]) # for multi-state

            next_state, reward, done, _ = env.step(action_list)
            state = next_state
            step_num += 1
            stop_cnt += 1
        env.bulk_log()
        write_summary(dic_path, task, self.dic_exp_conf["EPISODE_LEN"], batch_id, self.dic_traffic_env_conf['FLOW_FILE'])

    def reset_task(self, tasks, batch_id, reset_type='learning'):
        # regenerate new envs to avoid the engine stuck bug!
        dic_traffic_env_conf_list = []
        dic_path_list = []
        for task in tasks:
            dic_agent_conf = copy.deepcopy(self.dic_agent_conf)
            dic_agent_conf['TRAFFIC_FILE'] = task

            dic_traffic_env_conf = copy.deepcopy(self.task_traffic_env_map[task])
            dic_traffic_env_conf['TRAFFIC_FILE'] = task
            dic_traffic_env_conf_list.append(dic_traffic_env_conf)

            dic_path = copy.deepcopy(self.task_path_map[task])
            if reset_type == 'test':
                dic_path["PATH_TO_LOG"] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], reset_type + '_round',
                                                       task, 'tasks_round_' + str(batch_id))
            else:
                dic_path["PATH_TO_LOG"] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], reset_type+'_round',
                                                       'tasks_round_' + str(batch_id), task)
            dic_path_list.append(dic_path)

            if not os.path.exists(dic_path['PATH_TO_LOG']):
                os.makedirs(dic_path['PATH_TO_LOG'])

        self.envs = SubprocVecEnv(dic_path_list, dic_traffic_env_conf_list, len(tasks), queue=self.queue)