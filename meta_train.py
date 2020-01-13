from metalearner import MetaLearner
from sampler import BatchSampler
from multiprocessing import Process
import config
import time
import copy

import random
import numpy as np
import tensorflow as tf
import pickle
import shutil
from traffic import *
from utils import parse, config_all, parse_roadnet
import sys

def main(args):
    '''
        Perform meta-training for MAML and MetaLight

        Arguments:
            args: generated in utils.py:parse()
    '''

    # configuration: experiment, agent, traffic_env, path
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = config_all(args)

    _time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    postfix = args.postfix
    inner_memo = "_" + _time + postfix
    dic_traffic_env_conf["inner_memo"] = inner_memo
    dic_path.update({
        "PATH_TO_MODEL": os.path.join(dic_path["PATH_TO_MODEL"], inner_memo),
        "PATH_TO_WORK_DIRECTORY": os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], inner_memo),
        "PATH_TO_GRADIENT": os.path.join(dic_path["PATH_TO_GRADIENT"], inner_memo, "gradient"),
    })

    # traffic_env, traffic_category defined in traffic 
    dic_traffic_env_conf["TRAFFIC_IN_TASKS"] = traffic_category["train_all"]
    dic_traffic_env_conf["traffic_category"] = traffic_category

    random.seed(dic_agent_conf['SEED'])
    np.random.seed(dic_agent_conf['SEED'])
    tf.set_random_seed(dic_agent_conf['SEED'])

    # load or build initial model
    if not dic_agent_conf['PRE_TRAIN']:
        p = Process(target=build_init, args=(copy.deepcopy(dic_agent_conf),
                                             copy.deepcopy(dic_traffic_env_conf),
                                             copy.deepcopy(dic_path)))
        p.start()
        p.join()
    else:
        if not os.path.exists(dic_path['PATH_TO_MODEL']):
            os.makedirs(dic_path['PATH_TO_MODEL'])
        shutil.copy(os.path.join('model', 'initial', 'common', dic_agent_conf['PRE_TRAIN_MODEL_NAME'] + '.pkl'),
                    os.path.join(dic_path['PATH_TO_MODEL'], 'params' + "_" + "init.pkl"))

    for batch_id in range(args.run_round):
        # meta batch size process
        process_list = []
        task_num = min(len(dic_traffic_env_conf['TRAFFIC_IN_TASKS']), args.meta_batch_size)
        sample_task_traffic = random.sample(dic_traffic_env_conf['TRAFFIC_IN_TASKS'], task_num)
        if dic_traffic_env_conf["MODEL_NAME"] == "MetaLight":
            p = Process(target=metalight_train,
                        args=(copy.deepcopy(dic_exp_conf),
                              copy.deepcopy(dic_agent_conf),
                              copy.deepcopy(dic_traffic_env_conf),
                              copy.deepcopy(dic_path),
                              sample_task_traffic, batch_id)
                        )
            p.start()
            p.join()
        else: # maml
            for task in sample_task_traffic:
                p = Process(target=maml_train,
                            args=(copy.deepcopy(dic_exp_conf),
                                  copy.deepcopy(dic_agent_conf),
                                  copy.deepcopy(dic_traffic_env_conf),
                                  copy.deepcopy(dic_path),
                                  task, batch_id)
                            )
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()

            if not dic_traffic_env_conf['FIRST_PART']:
                meta_step(dic_path, dic_agent_conf, dic_traffic_env_conf, batch_id)

        ## update the epsilon
        decayed_epsilon = dic_agent_conf["EPSILON"] * pow(dic_agent_conf["EPSILON_DECAY"], batch_id)
        dic_agent_conf["EPSILON"] = max(decayed_epsilon, dic_agent_conf["MIN_EPSILON"])


def build_init(dic_agent_conf, dic_traffic_env_conf, dic_path):
    '''
        build initial model for maml and metalight

        Arguments:
            dic_agent_conf:         configuration of agent
            dic_traffic_env_conf:   configuration of traffic environment
            dic_path:               path of source files and output files
    '''

    any_task = dic_traffic_env_conf["traffic_category"]["train_all"][0]
    dic_traffic_env_conf["ROADNET_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][any_task][2]
    dic_traffic_env_conf["FLOW_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][any_task][3]
    # parse roadnet
    roadnet_path = os.path.join(dic_path['PATH_TO_DATA'], any_task.split(".")[0], dic_traffic_env_conf["traffic_category"]["traffic_info"][any_task][2])  # dic_traffic_env_conf['ROADNET_FILE'])
    lane_phase_info = parse_roadnet(roadnet_path)
    dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
    dic_traffic_env_conf["num_lanes"] = int(
        len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4)  # num_lanes per direction
    dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

    policy = config.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path
    )
    params = policy.init_params()
    if not os.path.exists(dic_path["PATH_TO_MODEL"]):
        os.makedirs(dic_path["PATH_TO_MODEL"])
    pickle.dump(params, open(os.path.join(dic_path['PATH_TO_MODEL'], 'params' + "_" + "init.pkl"), 'wb'))


def maml_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path, task, batch_id):
    '''
        maml meta-train function 

        Arguments:
            dic_exp_conf:           dict,   configuration of this experiment
            dic_agent_conf:         dict,   configuration of agent
            dic_traffic_env_conf:   dict,   configuration of traffic environment
            dic_path:               dict,   path of source files and output files
            task:                   string, traffic files name 
            batch_id:               int,    round number
    '''
    dic_path.update({
        "PATH_TO_DATA": os.path.join(dic_path['PATH_TO_DATA'], task.split(".")[0])
    })
    # parse roadnet
    dic_traffic_env_conf["ROADNET_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][2]
    dic_traffic_env_conf["FLOW_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][3]
    roadnet_path = os.path.join(dic_path['PATH_TO_DATA'],
                                dic_traffic_env_conf["traffic_category"]["traffic_info"][task][2])  # dic_traffic_env_conf['ROADNET_FILE'])
    lane_phase_info = parse_roadnet(roadnet_path)
    dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
    dic_traffic_env_conf["num_lanes"] = int(
        len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4)  # num_lanes per direction
    dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

    dic_traffic_env_conf["TRAFFIC_FILE"] = task

    sampler = BatchSampler(dic_exp_conf=dic_exp_conf,
                           dic_agent_conf=dic_agent_conf,
                           dic_traffic_env_conf=dic_traffic_env_conf,
                           dic_path=dic_path,
                           batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)

    policy = config.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path
    )

    metalearner = MetaLearner(sampler, policy,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              dic_path=dic_path
                              )

    if batch_id == 0:
        params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_init.pkl'), 'rb'))
        metalearner.meta_params = params
        metalearner.meta_target_params = params

    else:
        params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl' % (batch_id - 1)), 'rb'))
        metalearner.meta_params = params
        period = dic_agent_conf['PERIOD']
        target_id = int((batch_id - 1)/ period)
        metalearner.meta_target_params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl' % (target_id * period)), 'rb'))

    metalearner.sample_maml(task, batch_id)
    sys.exit()

def metalight_train(dic_exp_conf, dic_agent_conf, _dic_traffic_env_conf, _dic_path, tasks, batch_id):
    '''
        metalight meta-train function 

        Arguments:
            dic_exp_conf:           dict,   configuration of this experiment
            dic_agent_conf:         dict,   configuration of agent
            _dic_traffic_env_conf:  dict,   configuration of traffic environment
            _dic_path:              dict,   path of source files and output files
            tasks:                  list,   traffic files name in this round 
            batch_id:               int,    round number
    '''
    tot_path = []
    tot_traffic_env = []
    for task in tasks:
        dic_traffic_env_conf = copy.deepcopy(_dic_traffic_env_conf)
        dic_path = copy.deepcopy(_dic_path)
        dic_path.update({
            "PATH_TO_DATA": os.path.join(dic_path['PATH_TO_DATA'], task.split(".")[0])
        })
        # parse roadnet
        dic_traffic_env_conf["ROADNET_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][2]
        dic_traffic_env_conf["FLOW_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][3]
        roadnet_path = os.path.join(dic_path['PATH_TO_DATA'],
                                    dic_traffic_env_conf["traffic_category"]["traffic_info"][task][2])  # dic_traffic_env_conf['ROADNET_FILE'])
        lane_phase_info = parse_roadnet(roadnet_path)
        dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
        dic_traffic_env_conf["num_lanes"] = int(
            len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4)  # num_lanes per direction
        dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

        dic_traffic_env_conf["TRAFFIC_FILE"] = task

        tot_path.append(dic_path)
        tot_traffic_env.append(dic_traffic_env_conf)

    sampler = BatchSampler(dic_exp_conf=dic_exp_conf,
                           dic_agent_conf=dic_agent_conf,
                           dic_traffic_env_conf=tot_traffic_env,
                           dic_path=tot_path,
                           batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)

    policy = config.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=tot_traffic_env,
        dic_path=tot_path
    )

    metalearner = MetaLearner(sampler, policy,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=tot_traffic_env,
                              dic_path=tot_path
                              )

    if batch_id == 0:
        params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_init.pkl'), 'rb'))
        params = [params] * len(policy.policy_inter)
        metalearner.meta_params = params
        metalearner.meta_target_params = params

    else:
        params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl' % (batch_id - 1)), 'rb'))
        params = [params] * len(policy.policy_inter)
        metalearner.meta_params = params
        period = dic_agent_conf['PERIOD']
        target_id = int((batch_id - 1)/ period)
        meta_params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl' % (target_id * period)), 'rb'))
        meta_params = [meta_params] * len(policy.policy_inter)
        metalearner.meta_target_params = meta_params

    metalearner.sample_metalight(tasks, batch_id)

def meta_step(dic_path, dic_agent_conf, dic_traffic_env_conf, batch_id):
    '''
        update the common model's parameters of metalight 

        Arguments:
            dic_agent_conf:     dict,   configuration of agent
            dic_traffic_env_conf:   dict,   configuration of traffic environment
            dic_path:           dict,   path of source files and output files
            batch_id:           int,    round number
    '''
    grads = []
    try:
        f = open(os.path.join(dic_path['PATH_TO_GRADIENT'], "gradients_%d.pkl") % batch_id, "rb")
        while True:
            grads.append(pickle.load(f))
    except:
        pass
    if batch_id == 0:
        meta_params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_init.pkl'), 'rb'))
    else:
        meta_params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl'%(batch_id-1)), 'rb'))
    tot_grads = dict(zip(meta_params.keys(), [0] * len(meta_params.keys())))
    for key in meta_params.keys():
        for g in grads:
            tot_grads[key] += g[key]
    _beta = dic_agent_conf['BETA']
    meta_params = dict(zip(meta_params.keys(),
                    [meta_params[key] - _beta * tot_grads[key] for key in meta_params.keys()]))

    # save the meta parameters
    pickle.dump(meta_params,
                open(os.path.join(dic_path['PATH_TO_MODEL'], 'params' + "_" + str(batch_id) + ".pkl"), 'wb'))

if __name__ == '__main__':
    import os
    args = parse() # defined in utils.py
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    main(args)
