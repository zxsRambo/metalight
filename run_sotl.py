from metalearner import MetaLearner
from sampler import BatchSampler

import config
import time
import copy
from multiprocessing import Process
import pickle
import random
import numpy as np
import tensorflow as tf
from utils import parse, config_all, parse_roadnet
import os
from copy import deepcopy as dp
from traffic import *


def main(args):
    """
        Mainly run SOTL
    """
    ### *** exp, agent, traffic_env, path_conf
    _dic_exp_conf, _dic_agent_conf, _dic_traffic_env_conf, _dic_path = config_all(args)

    traffic_file_list = _dic_traffic_env_conf["TRAFFIC_CATEGORY"][args.traffic_group]
    process_list = []
    _dic_traffic_env_conf["FAST_BATCH_SIZE"] = 1

    for traffic_file in traffic_file_list:
        dic_exp_conf = dp(_dic_exp_conf)
        dic_agent_conf = dp(_dic_agent_conf)
        dic_traffic_env_conf = dp(_dic_traffic_env_conf)
        dic_path = dp(_dic_path)

        traffic_of_tasks = [traffic_file]

        dic_traffic_env_conf['ROADNET_FILE'] = dic_traffic_env_conf["TRAFFIC_CATEGORY"]["traffic_info"][traffic_file][2]
        dic_traffic_env_conf['FLOW_FILE'] = dic_traffic_env_conf["TRAFFIC_CATEGORY"]["traffic_info"][traffic_file][3]

        # path
        _time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
        postfix = ""
        dic_path.update({
            "PATH_TO_MODEL": os.path.join(dic_path["PATH_TO_MODEL"], traffic_file + "_" + _time + postfix),
            "PATH_TO_WORK_DIRECTORY": os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"],
                                                   traffic_file + "_" + _time + postfix),
            "PATH_TO_GRADIENT": os.path.join(dic_path["PATH_TO_GRADIENT"], traffic_file + "_" + _time + postfix,
                                             "gradient"),
            "PATH_TO_DATA": os.path.join(dic_path["PATH_TO_DATA"], traffic_file.split(".")[0])
        })
        # traffic env
        dic_traffic_env_conf["TRAFFIC_FILE"] = traffic_file
        dic_traffic_env_conf["TRAFFIC_IN_TASKS"] = [traffic_file]
        # parse roadnet
        roadnet_path = os.path.join(dic_path['PATH_TO_DATA'], dic_traffic_env_conf['ROADNET_FILE'])
        lane_phase_info = parse_roadnet(roadnet_path)
        dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
        dic_traffic_env_conf["num_lanes"] = int(len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4) # num_lanes per direction
        dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

        dic_exp_conf.update({
            "TRAFFIC_FILE": traffic_file,  
            "TRAFFIC_IN_TASKS": traffic_of_tasks})

        single_process = args.single_process
        if single_process:
            _train(copy.deepcopy(dic_exp_conf),
                   copy.deepcopy(dic_agent_conf),
                   copy.deepcopy(dic_traffic_env_conf),
                   copy.deepcopy(deploy_dic_path))
        else:
            p = Process(target=_train, args=(copy.deepcopy(dic_exp_conf),
                                             copy.deepcopy(dic_agent_conf),
                                             copy.deepcopy(dic_traffic_env_conf),
                                             copy.deepcopy(dic_path)))

            process_list.append(p)

    num_process = args.num_process
    if not single_process:
        i = 0
        list_cur_p = []
        for p in process_list:
            if len(list_cur_p) < num_process:
                print(i)
                p.start()
                list_cur_p.append(p)
                i += 1
            if len(list_cur_p) < num_process:
                continue

            idle = check_all_workers_working(list_cur_p)

            while idle == -1:
                time.sleep(1)
                idle = check_all_workers_working(
                    list_cur_p)
            del list_cur_p[idle]

        for i in range(len(list_cur_p)):
            p = list_cur_p[i]
            p.join()

def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1

def _train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):

    random.seed(dic_agent_conf['SEED'])
    np.random.seed(dic_agent_conf['SEED'])
    tf.set_random_seed(dic_agent_conf['SEED'])

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

    sampler.reset_task([dic_traffic_env_conf["TRAFFIC_FILE"]], 0, reset_type='test')
    sampler.sample_sotl(policy, dic_traffic_env_conf["TRAFFIC_FILE"])


if __name__ == '__main__':

    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    main(args)
