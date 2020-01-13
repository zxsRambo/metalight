import json
import config
import copy
import os
import pickle as pkl
from cityflow_env import CityFlowEnv

class Player(object):

    def __init__(self, path, scenario):
        self.work_path = "records/" + path
        self.dic_exp_conf = json.load(open(os.path.join(self.work_path, 'exp.conf')))
        self.dic_agent_conf = json.load(open(os.path.join(self.work_path, 'agent.conf')))
        self.dic_traffic_env_conf = json.load(open(os.path.join(self.work_path, 'traffic_env.conf')))
        # change key from str to int, due to json load
        str_int_key = ['phase_startLane_mapping', 'phase_sameStartLane_mapping', 'phase_roadLink_mapping']
        for _key in str_int_key:
            t = self.dic_traffic_env_conf["LANE_PHASE_INFO"][_key]
            self.dic_traffic_env_conf["LANE_PHASE_INFO"][_key] = { int(k): t[k] for k in t.keys()}
        # change dict to list
        for _key in self.dic_traffic_env_conf["LANE_PHASE_INFO"]['phase_roadLink_mapping'].keys():
            t = self.dic_traffic_env_conf["LANE_PHASE_INFO"]['phase_roadLink_mapping'][_key]
            self.dic_traffic_env_conf["LANE_PHASE_INFO"]['phase_roadLink_mapping'][_key] = [
                tuple(l) for l in t
            ]
        self.model_path = 'model/' + path
        self.data_path = "data/scenario/" + scenario

    def play(self, round, task, if_gui=False):
        dic_traffic_env_conf = copy.deepcopy(self.dic_traffic_env_conf)
        dic_traffic_env_conf['TRAFFIC_FILE'] = task
        if if_gui:
            dic_traffic_env_conf['SAVEREPLAY'] = True

        env = CityFlowEnv(path_to_log=os.path.join(self.work_path, 'test_round'),
                      path_to_work_directory=self.data_path,
                      dic_traffic_env_conf=dic_traffic_env_conf)

        policy = config.DIC_AGENTS[self.dic_exp_conf['MODEL_NAME']](
            dic_agent_conf=self.dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=None
        )

        params = pkl.load(open(os.path.join(self.model_path, 'params_%d.pkl'%round), 'rb'))
        policy.load_params(params)
        done = False
        state = env.reset()
        step_num = 0
        stop_cnt = 0
        while not done and step_num < int(
                self.dic_exp_conf["EPISODE_LEN"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            for one_state in state:
                action = policy.choose_action([[one_state]],
                                              test=True)  # one for multi-state, the other for multi-intersection
                action_list.append(action[0])  # for multi-state

            next_state, reward, done, _ = env.step(action_list)

            state = next_state
            step_num += 1
            stop_cnt += 1
        env.bulk_log()
        #self.write_summary(dic_path, 'task_%d_%s' % (task_id, task), self.dic_exp_conf["EPISODE_LEN"], batch_id)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rel_path = "memo/the_traffic_you_want_to_reply" # the path in "meta_train"
    scenario = rel_path.split("/")[-1].split(".json")[0] # traffic can just stand for scenario

    round_list = [197] # the model you want to replay
    task_list = [
        scenario
                 ]
    player = Player(rel_path, scenario)
    for round in round_list:
        for task in task_list:
            player.play(round, task, if_gui=True)