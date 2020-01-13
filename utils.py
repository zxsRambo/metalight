# collect the common function
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from math import isnan
import config
from traffic import *
import copy
import json
from collections import OrderedDict

def get_total_traffic_volume(traffic_file):
    # only support "cross" and "synthetic"
    if "cross" in traffic_file:
        sta = traffic_file.find("equal_") + len("equal_")
        end = traffic_file.find(".xml")
        return int(traffic_file[sta:end]) * 4

    elif "synthetic" in traffic_file:
        sta = traffic_file.rfind("-") + 1
        end = traffic_file.rfind(".json")
        return int(traffic_file[sta:end])

    elif "flow" in traffic_file:
        sta = traffic_file.find("flow_1_1_") + len("flow_1_1_")
        end = traffic_file.find(".json")
        return int(traffic_file[sta:end]) * 4

    elif "real" in traffic_file:
        sta = traffic_file.rfind("-") + 1
        end = traffic_file.rfind(".json")
        return int(traffic_file[sta:end])

    elif "hangzhou" in traffic_file:
        traffic = traffic_file.split(".json")[0]
        vol = int(traffic.split("_")[-1])
        return vol
    elif "ngsim" in traffic_file:
        traffic = traffic_file.split(".json")[0]
        vol = int(traffic.split("_")[-1])
        return vol

def write_summary(dic_path, test_traffic, episode_len, cnt_round, flow_file="flow.json"):
    record_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", test_traffic,
                              "tasks_round_" + str(cnt_round))
    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", test_traffic, "test_results.csv")
    path_to_seg_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", "test_seg_results.csv")
    num_seg = episode_len // 3600

    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    if not os.path.exists(path_to_log):
        df_col = pd.DataFrame(columns=("round", "duration", "vec_in", "vec_out"))
        if num_seg > 1:
            list_seg_col = ["round"]
            for i in range(num_seg):
                list_seg_col.append("duration-" + str(i))
            df_seg_col = pd.DataFrame(columns=list_seg_col)
            df_seg_col.to_csv(path_to_seg_log, mode="a", index=False)
        df_col.to_csv(path_to_log, mode="a", index=False)

    # summary items (duration) from csv
    df_vehicle_inter_0 = pd.read_csv(os.path.join(record_dir, "vehicle_inter_0.csv"),
                                     sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                     names=["vehicle_id", "enter_time", "leave_time"])

    vehicle_in = sum([int(x) for x in (df_vehicle_inter_0["enter_time"].values > 0)])
    vehicle_out = sum([int(x) for x in (df_vehicle_inter_0["leave_time"].values > 0)])
    duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
    ave_duration = np.mean([time for time in duration if not isnan(time)])
    # ********* new calculation of duration **********
    df_vehicle_planed_enter = get_planed_entering(os.path.join(dic_path["PATH_TO_DATA"], flow_file), episode_len)
    ave_duration = cal_travel_time(df_vehicle_inter_0, df_vehicle_planed_enter, episode_len)

    summary = {"round": [cnt_round], "duration": [ave_duration], "vec_in": [vehicle_in], "vec_out": [vehicle_out]}

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(path_to_log, mode="a", header=False, index=False)

    if num_seg > 1:
        list_duration_seg = [float('inf')] * num_seg
        nan_thres = 120
        for i, interval in enumerate(range(0, episode_len, 3600)):
            did = np.bitwise_and(df_vehicle_inter_0["enter_time"].values < interval + 3600,
                                 df_vehicle_inter_0["enter_time"].values > interval)
            duration_seg = df_vehicle_inter_0["leave_time"][did].values - df_vehicle_inter_0["enter_time"][
                did].values
            ave_duration_seg = np.mean([time for time in duration_seg if not isnan(time)])
            # print(traffic_file, round, i, ave_duration)
            real_traffic_vol_seg = 0
            nan_num_seg = 0
            for time in duration_seg:
                if not isnan(time):
                    real_traffic_vol_seg += 1
                else:
                    nan_num_seg += 1

            if nan_num_seg < nan_thres:
                list_duration_seg[i] = ave_duration_seg

        round_summary = {"round": [cnt_round]}
        for j in range(num_seg):
            key = "duration-" + str(j)
            if key not in round_summary.keys():
                round_summary[key] = [list_duration_seg[j]]
        round_summary = pd.DataFrame(round_summary)
        round_summary.to_csv(path_to_seg_log, mode="a", index=False, header=False)

def get_planed_entering(flowFile, episode_len):
    # todo--check with huichu about how each vehicle is inserted, according to the interval. 1s error may occur.
    list_flow = json.load(open(flowFile, "r"))
    dic_traj = {'vehicle_id':[], 'planed_enter_time':[]}
    for flow_id, flow in enumerate(list_flow):
        list_ts_this_flow = []
        for step in range(flow["startTime"], min(flow["endTime"] + 1, episode_len)):
            if step == flow["startTime"]:
                list_ts_this_flow.append(step)
            elif step - list_ts_this_flow[-1] >= flow["interval"]:
                list_ts_this_flow.append(step)

        for vec_id, ts in enumerate(list_ts_this_flow):
            dic_traj['vehicle_id'].append("flow_{0}_{1}".format(flow_id, vec_id))
            dic_traj['planed_enter_time'].append(ts)
            #dic_traj["flow_{0}_{1}".format(flow_id, vec_id)] = {"planed_enter_time": ts}

    df = pd.DataFrame(dic_traj)
    #df.set_index('vehicle_id')
    return df
    #return pd.DataFrame(dic_traj).transpose()

def cal_travel_time(df_vehicle_actual_enter_leave, df_vehicle_planed_enter, episode_len):
    df_vehicle_planed_enter.set_index('vehicle_id', inplace=True)
    df_vehicle_actual_enter_leave.set_index('vehicle_id', inplace=True)
    df_res = pd.concat([df_vehicle_planed_enter, df_vehicle_actual_enter_leave], axis=1, sort=False)
    assert len(df_res) == len(df_vehicle_planed_enter)

    df_res["leave_time"].fillna(episode_len, inplace=True)
    df_res["travel_time"] = df_res["leave_time"] - df_res["planed_enter_time"]
    travel_time = df_res["travel_time"].mean()
    return travel_time

def parse():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Meta RLSignal')

    parser.add_argument("--memo", type=str, default="default")
    parser.add_argument("--algorithm", type=str, default="MetaLight")

    parser.add_argument("--num_phase", type=int, default=4)
    parser.add_argument("--norm", type=str, default='None')

    parser.add_argument('--num_updates', type=int, default=1,
                        help='number of step for fast learning')
    parser.add_argument('--num_gradient_step', type=int, default=1,
                        help='number of gradient step when updating para')
    parser.add_argument('--period', type=int, default=5,
                        help='to update the target dqn')

    parser.add_argument('--activation_fn', type=str, default='leaky_relu')
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=11)

    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--single_process", action='store_true')
    parser.add_argument("--num_process", type=int, default=2)
    #parser.add_argument("--sample_size", type=int, default=1440)
    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--min_epsilon", type=float, default=0.2)

    parser.add_argument("--reward_norm", action="store_true")
    parser.add_argument("--input_norm", action='store_true')

    parser.add_argument("--first_part", action="store_true")
    parser.add_argument("--gradient_clip", action="store_true")
    parser.add_argument("--clip_size", type=float, default=1)
    parser.add_argument("--pre_train", action="store_true")
    parser.add_argument("--pre_train_model_name", type=str, default='self')

    parser.add_argument('--fast_batch_size', type=int, default=1,
                        help='batch size for each individual task')

    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers for trajectories sampling')

    parser.add_argument('--meta_batch_size', type=int, default=2,
                        help='number of tasks per batch')

    parser.add_argument("--env", type=str, default="traffic",
                        help='traffic and point for choice')

    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--episode_len", type=int, default=3600)
    parser.add_argument("--run_round", type=int, default=100)
    parser.add_argument("--test_episode_len", type=int, default=3600)
    parser.add_argument("--done", action="store_true")

    parser.add_argument("--change_path", action="store_true")

    parser.add_argument("--visible_gpu", type=str, default="")
    parser.add_argument("--multi_episodes", action="store_true")

    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--update_start", type=int, default=100)
    parser.add_argument("--update_period", type=int, default=1)
    parser.add_argument("--learning_rate_decay_step", type=int, default=100)
    parser.add_argument("--min_alpha", type=float, default=0.001)
    parser.add_argument("--test_period", type=int, default=90)
    parser.add_argument("--roadnet", type=str, default="roadnet_p4a_lt.json")
    parser.add_argument("--flow_file", type=str, default="flow.json")

    # for main
    parser.add_argument("--postfix", type=str, default='')
    parser.add_argument("--meta_update_period", type=int, default=10)
    parser.add_argument("--traffic_group", type=str, default="train_all")

    parser.add_argument("--city_test", action="store_true")
    args = parser.parse_args()
    return args

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result

def config_all(args):
    dic_traffic_env_conf_extra = {
        # file
        "ROADNET_FILE": args.roadnet,
        "FLOW_FILE": args.flow_file,

        # gui
        "SAVEREPLAY": args.replay,

        "EPISODE_LEN": args.episode_len,

        "DONE_ENABLE": args.done,

        # different env (traffic or point)


        # normalization
        "REWARD_NORM": args.reward_norm,
        "INPUT_NORM": args.input_norm,

        "FIRST_PART": False,

        "NUM_ROW": 1,
        "NUM_COL": 1,

        # state & reward
        "LIST_STATE_FEATURE": [ "cur_phase", "lane_num_vehicle"],



        "DIC_REWARD_INFO": {"sum_num_vehicle_been_stopped_thres1": -0.25},

        "LANE_NUM": {
            "LEFT": 1,
            "RIGHT": 0,
            "STRAIGHT": 1
        },

        "PHASE": [
            'WT_ET',
            'NT_ST',
            'WL_EL',
            'NL_SL',
            # 'WT_WL',
            # 'ET_EL',
            # 'NT_NL',
            # 'ST_SL',
        ],

        "LOG_DEBUG": args.debug,
        "FAST_BATCH_SIZE": args.fast_batch_size,
        'MODEL_NAME': args.algorithm,
        "TRAFFIC_CATEGORY": traffic_category,
    }

    if args.algorithm == "SOTL":
        dic_traffic_env_conf_extra["LIST_STATE_FEATURE"] = [
            "cur_phase_index",
            "time_this_phase",

            "lane_num_vehicle",
            "lane_num_vehicle_been_stopped_thres1",
        ]

    if args.num_phase == 2:
        dic_traffic_env_conf_extra.update(
            {
                "LANE_NUM": {
                    "LEFT": 0,
                    "RIGHT": 0,
                    "STRAIGHT": 1
                },

                "PHASE": [
                    'WT_ET',
                    'NT_ST',
                    # 'WL_EL',
                    # 'NL_SL',

                ]
            }
        )
    elif args.num_phase == 4:
        dic_traffic_env_conf_extra.update(
            {
                "LANE_NUM": {
                    "LEFT": 1,
                    "RIGHT": 0,
                    "STRAIGHT": 1
                },

                "PHASE": [
                    'WT_ET',
                    'NT_ST',
                    'WL_EL',
                    'NL_SL',

                ]
            }
        )
    elif args.num_phase == 6:
        dic_traffic_env_conf_extra.update(
            {
                "LANE_NUM": {
                    "LEFT": 1,
                    "RIGHT": 0,
                    "STRAIGHT": 1
                },

                "PHASE": [
                    'WT_ET',
                    'NT_ST',
                    'WL_EL',
                    'NL_SL',
                    'WT_WL',
                    'ET_EL',
                ]
            }
        )
    elif args.num_phase == 8:
        dic_traffic_env_conf_extra.update(
            {
                "LANE_NUM": {
                    "LEFT": 1,
                    "RIGHT": 0,
                    "STRAIGHT": 1
                },

                "PHASE": [
                    'WT_ET',
                    'NT_ST',
                    'WL_EL',
                    'NL_SL',
                    'WT_WL',
                    'ET_EL',
                    'ST_SL',
                    'NT_NL',

                ]
            }
        )

    # policy & agent config
    dic_agent_conf_extra = {
        "UPDATE_Q_BAR_FREQ": 5,
        # network

        "N_LAYER": 2,
        'NORM': args.norm,
        'NUM_UPDATES': args.num_updates,
        'NUM_GRADIENT_STEP': args.num_gradient_step,

        'PERIOD': args.period,
        'ACTIVATION_FN': args.activation_fn,
        'GRADIENT_CLIP': True,
        'CLIP_SIZE': args.clip_size,
        'PRE_TRAIN': args.pre_train,
        'PRE_TRAIN_MODEL_NAME': args.pre_train_model_name,

        'OPTIMIZER': args.optimizer,

        #
        "ALPHA": args.alpha,
        "BETA": args.beta,
        'SEED': args.seed,

        "EPSILON": args.epsilon,
        "MIN_EPSILON": args.min_epsilon,

        "MULTI_EPISODES": args.multi_episodes,

        #
        'SAMPLE_SIZE': args.sample_size,
        'UPDATE_START': args.update_start,  # 500,
        'UPDATE_PERIOD': args.update_period,  # 300,
        "TEST_PERIOD": args.test_period,
        "ALPHA_DECAY_STEP": args.learning_rate_decay_step,
        "MIN_ALPHA": args.min_alpha,

        "META_UPDATE_PERIOD": args.meta_update_period,
    }

    # path config
    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join("model", args.memo),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", args.memo),
        "PATH_TO_DATA": os.path.join("data", "scenario"),
        "PATH_TO_ERROR": os.path.join("errors", args.memo),
        "PATH_TO_GRADIENT": os.path.join("records", args.memo),
    }

    # experiment config
    dic_exp_conf_extra = {
        "EPISODE_LEN": args.episode_len,
        "TEST_EPISODE_LEN": args.test_episode_len,
        "MODEL_NAME": args.algorithm,  # Todo

        "NUM_ROUNDS": args.run_round,
        "NUM_GENERATORS": 3,

        "NUM_EPISODE": 1,

        "MODEL_POOL": False,
        "NUM_BEST_MODEL": 1,

        "PRETRAIN": False,
        "PRETRAIN_NUM_ROUNDS": 20,
        "PRETRAIN_NUM_GENERATORS": 15,

        "AGGREGATE": False,
        "DEBUG": False,
        "EARLY_STOP": False,
    }

    model_name = args.algorithm
    deploy_dic_exp_conf = merge(config.DIC_EXP_CONF, dic_exp_conf_extra)
    deploy_dic_agent_conf = merge(getattr(config, "DIC_{0}_AGENT_CONF".format(model_name.upper())),
                                  dic_agent_conf_extra)
    deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
    deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)
    return deploy_dic_exp_conf, deploy_dic_agent_conf, deploy_dic_traffic_env_conf, deploy_dic_path

def parse_roadnet(roadnetFile):
    roadnet = json.load(open(roadnetFile))
    lane_phase_info_dict = OrderedDict()

    common_ids = [inter["id"] for inter in roadnet["intersections"] if not inter["virtual"]]
    atlanta_ids = ['69421277',
                   #'69249210',
                   '69387071', "69227168", "69515842"]
    la_ids = ['361142995', '2200742494', 'cluster_16298125_4757166089', '269390046']

    # many intersections exist in the roadnet and virtual intersection is controlled by signal
    if "la" in roadnetFile and "atlanta" not in roadnetFile:
        inter_ids = la_ids
    elif "atlanta" in roadnetFile:
        inter_ids = atlanta_ids
    else:
        inter_ids = common_ids

    intersections = []
    for id in inter_ids:
        for inter in roadnet["intersections"]:
            if inter['id'] == id:
                intersections.append(inter)

    for intersection in intersections:
        lane_phase_info_dict[intersection['id']] = {"start_lane": [],
                                                    "same_start_lane": [],
                                                     "end_lane": [],
                                                     "phase": [],
                                                     "yellow_phase": 0,
                                                     "phase_startLane_mapping": {},
                                                     "phase_noRightStartLane_mapping": {},
                                                     "phase_sameStartLane_mapping": {},
                                                     "phase_roadLink_mapping": {}}
        road_links = intersection["roadLinks"]

        start_lane = []
        same_start_lane = []
        end_lane = []
        roadLink_lane_pair = {ri: [] for ri in
                              range(len(road_links))}  # roadLink includes some lane_pair: (start_lane, end_lane)
        roadLink_same_start_lane = {ri: [] for ri in
                              range(len(road_links))}  # roadLink includes some lane_pair: (start_lane, end_lane)

        for ri in range(len(road_links)):
            road_link = road_links[ri]
            tmp_same_start_lane = []
            for lane_link in road_link["laneLinks"]:
                sl = road_link['startRoad'] + "_" + str(lane_link["startLaneIndex"])
                el = road_link['endRoad'] + "_" + str(lane_link["endLaneIndex"])
                type = road_link['type']
                start_lane.append(sl)
                tmp_same_start_lane.append(sl)
                end_lane.append(el)
                roadLink_lane_pair[ri].append((sl, el, type))
            tmp_same_start_lane = tuple(set(tmp_same_start_lane))
            roadLink_same_start_lane[ri].append(tmp_same_start_lane)
            same_start_lane.append(tmp_same_start_lane)


        lane_phase_info_dict[intersection['id']]["start_lane"] = sorted(list(set(start_lane)))
        lane_phase_info_dict[intersection['id']]["end_lane"] = sorted(list(set(end_lane)))
        lane_phase_info_dict[intersection['id']]["same_start_lane"] = sorted(list(set(same_start_lane)))

        for phase_i in range(len(intersection["trafficLight"]["lightphases"])):
            if len(intersection["trafficLight"]["lightphases"][phase_i]["availableRoadLinks"]) == 0:
                lane_phase_info_dict[intersection['id']]["yellow_phase"] = phase_i
                continue
            p = intersection["trafficLight"]["lightphases"][phase_i]
            lane_pair = []
            start_lane = []
            same_start_lane = []
            no_right_start_lane = []
            for ri in p["availableRoadLinks"]:
                for i in range(len(roadLink_lane_pair[ri])):
                    if roadLink_lane_pair[ri][i][0] not in start_lane:
                        start_lane.append(roadLink_lane_pair[ri][i][0])
                    if roadLink_lane_pair[ri][i][0] not in no_right_start_lane and roadLink_lane_pair[ri][i][2] != "turn_right":
                        no_right_start_lane.append(roadLink_lane_pair[ri][i][0])
                    if roadLink_lane_pair[ri][i][2] != "turn_right":
                        lane_pair.extend(roadLink_lane_pair[ri]) # no right roadlink

                if roadLink_same_start_lane[ri][0] not in same_start_lane:
                    same_start_lane.append(roadLink_same_start_lane[ri][0])
            lane_phase_info_dict[intersection['id']]["phase"].append(phase_i)
            lane_phase_info_dict[intersection['id']]["phase_startLane_mapping"][phase_i] = start_lane
            lane_phase_info_dict[intersection['id']]["phase_noRightStartLane_mapping"][phase_i] = no_right_start_lane
            lane_phase_info_dict[intersection['id']]["phase_sameStartLane_mapping"][phase_i] = same_start_lane
            lane_phase_info_dict[intersection['id']]["phase_roadLink_mapping"][phase_i] = list(set(lane_pair)) # tmp to remove repeated

    return lane_phase_info_dict


def contruct_layer(inp, activation_fn, reuse, norm, is_train, scope):
    if norm == 'batch_norm':
        out = tf.contrib.layers.batch_norm(inp, activation_fn=activation_fn,
                                           reuse=reuse, is_training=is_train,
                                           scope=scope)
    elif norm == 'None':
        out = activation_fn(inp)
    else:
        ValueError('Can\'t recognize {}'.format(norm))
    return out


def get_session(num_cpu):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1/10.
    tf_config.gpu_options.allow_growth = True
    return tf.Session(config=tf_config)
