import numpy as np
import multiprocessing as mp
import sys
is_py2 = (sys.version[0] == '2')
if is_py2:
    import Queue as queue
else:
    import queue as queue

sys.path.append('../..')
from cityflow_env import CityFlowEnv
import os

class EnvWorker(mp.Process):
    def __init__(self, remote, dic_path, dic_traffic_env_conf, queue, lock):
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.dic_path = dic_path
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.queue = queue
        self.lock = lock
        self.task_id = None
        self.done = False

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_LOG"],
            path_to_work_directory=self.dic_path["PATH_TO_DATA"],
            dic_traffic_env_conf=self.dic_traffic_env_conf)

    def empty_step(self):
        observation = [{'cur_phase': [0], 'lane_num_vehicle': [0, 0, 0, 0, 0, 0, 0, 0]}]
        reward, done = [0.0], True
        return observation, reward, done, []

    def try_reset(self):
        with self.lock:
            try:
                self.task_id = self.queue.get(True)
                self.done = (self.task_id is None)
            except queue.Empty:
                self.done = True
        if not self.done:
            if "test_round" not in self.dic_path['PATH_TO_LOG']:
                new_path_to_log = os.path.join(self.dic_path['PATH_TO_LOG'],
                                                            'episode_%d' % (self.task_id))
            else:
                new_path_to_log = self.dic_path['PATH_TO_LOG']
            self.env.modify_path_to_log(new_path_to_log)
            if not os.path.exists(new_path_to_log):
                os.makedirs(new_path_to_log)
            state = self.env.reset()
            #observation = (np.zeros(self.env.observation_space.shape,
            #    dtype=np.float32) if self.done else self.env.reset())
            return state
        else:
            return False

    def run(self):
        while True:
            command, data = self.remote.recv()
            if command == 'step':
                observation, reward, done, info = self.env.step(data)
                if done:
                    self.env.bulk_log()
                    self.try_reset()
                    #observation = self.try_reset()
                self.remote.send((observation, reward, done, self.task_id, info, self.done))
            elif command == 'reset':
                observation = self.try_reset()
                self.remote.send((observation, self.task_id))
            elif command == 'reset_task':
                self.env.unwrapped.reset_task(data)
                self.remote.send(True)
            elif command == 'close':
                self.remote.close()
                break
            elif command == 'get_spaces':
                self.remote.send((self.env.observation_space,
                                 self.env.action_space))
            elif command == 'bulk_log':
                self.env.bulk_log()
                self.remote.send(True)
            else:
                raise NotImplementedError()

class SubprocVecEnv():
    def __init__(self, dic_path_list, dic_traffic_env_conf_list, num_workers, queue):
        """
            Environment controller: single controller (agent) multiple environments
            Arguments:
                num_workers: number of environments (worker)
                queue:       process queue
        """
        self.lock = mp.Lock()
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_workers)])

        self.remotes = list(self.remotes)
        self.work_remotes = list(self.work_remotes)
        self.workers = [EnvWorker(self.work_remotes[i], dic_path_list[i], dic_traffic_env_conf_list[i], queue, self.lock)
                        for i in range(num_workers)]
        self.num_workers = num_workers
        for worker in self.workers:
            worker.daemon = True
            worker.start()

        #for remote in self.work_remotes:
        #    remote.close()
        self.waiting = False
        self.closed = False

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, task_ids, infos, worker_dones= zip(*results)

        #del_worker_id = []
        for worker_id in range(len(worker_dones) -1, -1, -1):
            if worker_dones[worker_id]:
                self.remotes[worker_id].send(('close', None))
                self.workers[worker_id].join()
                self.work_remotes[worker_id].close()
                self.remotes[worker_id].close()
                del self.remotes[worker_id]
                del self.work_remotes[worker_id]

                # del_worker_id.append(worker_id)
        #for worker_id in del_worker_id.reverse():
        #    del self.remotes[worker_id]

        return np.stack(observations), np.stack(rewards), np.stack(dones), task_ids, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        observations, task_ids = zip(*results)
        return np.stack(observations), task_ids

    def bulk_log(self):
        for remote in self.remotes:
            remote.send(('bulk_log', None))
        results = [remote.recv() for remote in self.remotes]

    def reset_task(self, tasks):
        for remote, task in zip(self.remotes, tasks):
            remote.send(('reset_task', task))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True
