import pickle
import numpy as np
import json
import sys
import pandas as pd
import os
import utils
import json

"""
    Class CityFlowEnv provides the environment for traffic signal control of single (or multiple) intersections
    Class Intersection specifies the environment for single intersection
"""
class Intersection:
    DIC_PHASE_MAP = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        -1: 0
    }
    def __init__(self, inter_id, dic_traffic_env_conf, eng):
        self.inter_id = inter_id
        self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])

        self.eng = eng
        self.dic_traffic_env_conf = dic_traffic_env_conf

        # =====  intersection settings =====
        self.list_entering_lanes = dic_traffic_env_conf["LANE_PHASE_INFO"]["start_lane"]
        self.list_exiting_lanes = dic_traffic_env_conf["LANE_PHASE_INFO"]["end_lane"]
        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        # grid settings
        self.length_lane = 300
        self.length_terminal = 50
        self.length_grid = 5
        self.num_grid = int(self.length_lane // self.length_grid)

        # previous & current
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}

        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step = []
        self.list_lane_vehicle_current_step = []

        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)

        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

        self.dic_vehicle_min_speed = {}  # this second
        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second

    def set_signal(self, action, action_pattern, yellow_time, all_red_time):
        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time: # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index) # if multi_phase, need more adjustment
                self.all_yellow_flag = False
            else:
                pass
        else:
            # determine phase
            if action_pattern == "switch": # switch by order
                if action == 0: # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1: # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.list_phases) # if multi_phase, need more adjustment
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set": # set to certain phase
                self.next_phase_to_set_index = action + 1  # !!! if multi_phase, need more adjustment

            # set phase
            if self.current_phase_index == self.next_phase_to_set_index: # the light phase keeps unchanged
                pass
            else: # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.eng.set_tl_phase(self.inter_name, 0) # !!! yellow, tmp
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index

        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step
        self.dic_lane_waiting_vehicle_count_previous_step = self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step

    def update_current_measurements(self):
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []

            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)

            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = self.eng.get_lane_vehicles()# not implement
        self.dic_lane_waiting_vehicle_count_current_step = self.eng.get_lane_waiting_vehicle_count()
        if self.dic_traffic_env_conf["LOG_DEBUG"]:
            self.dic_vehicle_speed_current_step = self.eng.get_vehicle_speed()
            self.dic_vehicle_distance_current_step = self.eng.get_vehicle_distance()

        # get vehicle list
        self.list_lane_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)

        list_vehicle_new_arrive = list(set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = list(set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))
        list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicle_new_left_entering_lane = []
        for l in list_vehicle_new_left_entering_lane_by_lane:
            list_vehicle_new_left_entering_lane += l

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left_entering_lane)

        # update vehicle minimum speed in history, # to be implemented
        #self._update_vehicle_min_speed()

        # update feature
        self._update_feature()

    def _update_leave_entering_approach_vehicle(self):
        list_entering_lane_vehicle_left = []
        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append(
                    list(
                        set(self.dic_lane_vehicle_previous_step[lane]) - \
                        set(self.dic_lane_vehicle_current_step[lane])
                    )
                )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicle_arrive):
        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            else:
                #print("vehicle: %s already exists in entering lane!"%vehicle)
                #sys.exit(-1)
                pass

    def _update_left_time(self, list_vehicle_left):
        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def _update_feature(self):
        dic_feature = dict()

        phase = [0] * len(self.list_entering_lanes)
        if self.current_phase_index != -1:
            start_lane = self.dic_traffic_env_conf["LANE_PHASE_INFO"]["phase_startLane_mapping"][self.current_phase_index]
            for lane in start_lane:
                phase[self.list_entering_lanes.index(lane)] = 1

        dic_feature["cur_phase"] = phase
        dic_feature["cur_phase_index"] = [self.current_phase_index]

        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["vehicle_position_img"] = None #self._get_lane_vehicle_position(self.list_entering_lanes)
        dic_feature["vehicle_speed_img"] = None #self._get_lane_vehicle_speed(self.list_entering_lanes)
        dic_feature["vehicle_acceleration_img"] = None
        dic_feature["vehicle_waiting_time_img"] = None #self._get_lane_vehicle_accumulated_waiting_time(self.list_entering_lanes)

        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres01"] = self._get_lane_num_vehicle_been_stopped(0.1, self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres1"] = self._get_lane_num_vehicle_been_stopped(1, self.list_entering_lanes)
        dic_feature["lane_queue_length"] = self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_left"] = None
        dic_feature["lane_sum_duration_vehicle_left"] = None
        dic_feature["lane_sum_waiting_time"] = None #self._get_lane_sum_waiting_time(self.list_entering_lanes)

        dic_feature["terminal"] = None

        self.dic_feature = dic_feature

    # ================= calculate features from current observations ======================
    def _get_lane_queue_length(self, list_lanes):
        '''
        queue length for each lane
        '''
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in list_lanes]

    def _get_lane_sum_waiting_time(self, list_lanes):
        '''
        waiting time for each lane
        '''
        raise NotImplementedError

    def _get_lane_list_vehicle_left(self, list_lanes):
        '''
        get list of vehicles left at each lane
        ####### need to check
        '''
        raise NotImplementedError

    def _get_lane_num_vehicle_left(self, list_lanes):
        list_lane_vehicle_left = self._get_lane_list_vehicle_left(list_lanes)
        list_lane_num_vehicle_left = [len(lane_vehicle_left) for lane_vehicle_left in list_lane_vehicle_left]
        return list_lane_num_vehicle_left

    def _get_lane_sum_duration_vehicle_left(self, list_lanes):
        ## not implemented error
        raise NotImplementedError

    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):
        if self.dic_traffic_env_conf['INPUT_NORM']:
            return [self.dic_lane_waiting_vehicle_count_current_step[lane] / 40 for lane in list_lanes]
        else:
            return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_position_grid_along_lane(self, vec):
        pos = int(self.dic_vehicle_sub_current_step[vec][get_traci_constant_mapping("VAR_LANEPOSITION")])
        return min(pos//self.length_grid, self.num_grid)

    def _get_lane_vehicle_position(self, list_lanes):
        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.zeros(self.num_grid)
            list_vec_id = self.dic_lane_vehicle_current_step[lane]
            for vec in list_vec_id:
                pos = int(self.dic_vehicle_distance_current_step[vec])
                pos_grid = min(pos//self.length_grid, self.num_grid)
                lane_vector[pos_grid] = 1
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)
    
    # debug
    def _get_vehicle_info(self, veh_id):
        try:
            pos = self.dic_vehicle_distance_current_step[veh_id]
            speed = self.dic_vehicle_speed_current_step[veh_id]
            return pos, speed
        except:
            return None, None

    def _get_lane_vehicle_speed(self, list_lanes):
        return [self.dic_vehicle_speed_current_step[lane] for lane in list_lanes]

    def _get_lane_vehicle_accumulated_waiting_time(self, list_lanes):
        raise NotImplementedError

    # ================= get functions from outside ======================
    def get_current_time(self):
        return self.eng.get_current_time()

    def get_dic_vehicle_arrive_leave_time(self):
        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):
        return self.dic_feature

    def get_state(self, list_state_features):
        dic_state = {state_feature_name: self.dic_feature[state_feature_name] for state_feature_name in list_state_features}
        return dic_state

    def get_reward(self, dic_reward_info):
        dic_reward = dict()
        dic_reward["flickering"] = None
        dic_reward["sum_lane_queue_length"] = None
        dic_reward["sum_lane_wait_time"] = None
        dic_reward["sum_lane_num_vehicle_left"] = None
        dic_reward["sum_duration_vehicle_left"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres01"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(self.dic_feature["lane_num_vehicle_been_stopped_thres1"])

        if self.dic_traffic_env_conf['REWARD_NORM']:
            # normalize the reward
            reward = - 2 * dic_reward["sum_num_vehicle_been_stopped_thres1"] / (40 * len(self.dic_feature["lane_num_vehicle_been_stopped_thres1"])) + 1
        else:
            reward = 0
            for r in dic_reward_info:
                if dic_reward_info[r] != 0:
                    reward += dic_reward_info[r] * dic_reward[r]
        return reward

class CityFlowEnv:
    list_intersection_id = [
        "intersection_1_1"
    ]

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        """
            The following commented codes shows the latest API of cityflow. 
            However, it will somehow hang in the multi-process. Therefore, another version of cityflow is used here, "engine.cpython-36m-x86_64-linux-gnu.so meta2:/metalight"
        """
        #import cityflow as engine
        #config_dict = {
        #    "interval": self.dic_traffic_env_conf["INTERVAL"],
        #    "seed": 0,
        #    "dir": "",
        #    "roadnetFile": os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf['ROADNET_FILE']),
        #    "flowFile": os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["FLOW_FILE"]),
        #    "rlTrafficLight": self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
        #    "saveReplay": self.dic_traffic_env_conf["SAVEREPLAY"],
        #    "roadnetLogFile": "frontend/web/testcase_roadnet_3x3.json",
        #    "replayLogFile": "frontend/web/testcase_replay_3x3.txt"
        #}
        #config_path = os.path.join(path_to_log, "cityflow_config")
        #with open(config_path, "w") as f:
        #    config_obj = json.dump(config_dict, f)
        #    print("dump cityflow config")
        #    print(config_path)
        #self.eng = engine.Engine(config_path, self.dic_traffic_env_conf["THREADNUM"])

        import engine
        self.eng = engine.Engine(self.dic_traffic_env_conf["INTERVAL"],
                                 self.dic_traffic_env_conf["THREADNUM"],
                                 self.dic_traffic_env_conf["SAVEREPLAY"],
                                 self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
                                 False)
        self.load_roadnet(self.dic_traffic_env_conf["ROADNET_FILE"])
        self.load_flow(self.dic_traffic_env_conf["FLOW_FILE"])

        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.stop_cnt = 0

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print ("MIN_ACTION_TIME should include YELLOW_TIME")
            pass
            #raise ValueError

    def modify_path_to_log(self, path_to_log):
        self.path_to_log = path_to_log

    def reset(self):
        self.eng.reset()

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i+1, j+1), self.dic_traffic_env_conf, self.eng) for i in range(self.dic_traffic_env_conf["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_conf["NUM_COL"])]
        self.list_inter_log = [[] for i in range(self.dic_traffic_env_conf["NUM_ROW"] *
                                                 self.dic_traffic_env_conf["NUM_COL"])]
        # get lanes list
        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        # get new measurements
        for inter in self.list_intersection:
            inter.update_current_measurements()

        state = self.get_state()
        return state

    def step(self, action):
        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        average_reward_action = 0
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            before_action_feature = self.get_feature()
            state = self.get_state()

            if self.dic_traffic_env_conf['DEBUG']:
                print("time: {0}, phase: {1}, time this phase: {2}, action: {3}".format(instant_time, before_action_feature[0]["cur_phase_index"], before_action_feature[0]["time_this_phase"], action_in_sec_display[0]))
            else:
                if i == 0:
                    #pass
                    print("time: {0}, phase: {1}, time this phase: {2}, action: {3}".format(instant_time,
                                                                                        before_action_feature[0][
                                                                                            "cur_phase_index"],
                                                                                        before_action_feature[0][
                                                                                            "time_this_phase"],
                                                                                        action_in_sec_display[0]))

            # _step
            self._inner_step(action_in_sec)

            # get reward
            reward = self.get_reward()
            average_reward_action = (average_reward_action*i + reward[0])/(i+1)

            # log
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)

            next_state = self.get_state()
            done = self._check_episode_done(next_state)

        return next_state, reward, done, [average_reward_action]

    def _inner_step(self, action):
        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()

        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                all_red_time=self.dic_traffic_env_conf["ALL_RED_TIME"]
            )

        # run one step
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()
        # get new measurements
        for inter in self.list_intersection:
            inter.update_current_measurements()

    def load_roadnet(self, roadnetFile=None):
        if not roadnetFile:
            roadnetFile = "roadnet_1_1.json"
        self.eng.load_roadnet(os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf['ROADNET_FILE']))
        print("successfully load roadnet: ", roadnetFile)

    def load_flow(self, flowFile=None):
        if not flowFile:
            flowFile = "flow_1_1.json"
        self.eng.load_flow(os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["FLOW_FILE"]))
        print("successfully load flowFile: ", self.dic_traffic_env_conf["FLOW_FILE"])

    def _check_episode_done(self, state):
        if self.get_current_time() >= self.dic_traffic_env_conf['EPISODE_LEN']:
            return True
        else:
            if self.dic_traffic_env_conf["DONE_ENABLE"]:
                if 39 in state[0]["lane_num_vehicle"]:
                    self.stop_cnt += 1

                if self.stop_cnt == 100:
                    self.stop_cnt = 0
                    return True
                else:
                    return False
            else:
                return False

    @staticmethod
    def convert_dic_to_df(dic):
        list_df = []
        for key in dic:
            df = pd.Series(dic[key], name=key)
            list_df.append(df)
        return pd.DataFrame(list_df)

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):
        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]

        return list_state

    def get_reward(self):
        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in self.list_intersection]

        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):
        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                    "state": before_action_feature[inter_ind],
                                                    "action": action[inter_ind]})

    def bulk_log(self):
        valid_flag = {}
        for inter_ind in range(len(self.list_intersection)):
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = self.convert_dic_to_df(dic_vehicle)
            df.to_csv(path_to_log_file, na_rep="nan")

            inter = self.list_intersection[inter_ind]
            feature = inter.get_feature()
            print(feature['lane_num_vehicle'])
            if max(feature['lane_num_vehicle']) > self.dic_traffic_env_conf["VALID_THRESHOLD"]:
                valid_flag[inter_ind] = 0
            else:
                valid_flag[inter_ind] = 1
        json.dump(valid_flag, open(os.path.join(self.path_to_log, "valid_flag.json"), "w"))

        #for inter_ind in range(len(self.list_inter_log)):
        #    path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
        #    f = open(path_to_log_file, "wb")
        #    pickle.dump(self.list_inter_log[inter_ind], f)
        #    f.close()
        self.log_replay()

    def log_replay(self):
        vol = utils.get_total_traffic_volume(self.dic_traffic_env_conf["TRAFFIC_FILE"])
        # self.eng.print_log(os.path.join("data", "frontend", "web", "roadnet_1_1.json"),
        #                         os.path.join("data", "frontend", "web", "replay_1_1_%s.txt"%vol))
        self.eng.print_log(os.path.join(self.path_to_log, "roadnet_%s.json" % vol),
                           os.path.join(self.path_to_log, "replay_%s.txt" % vol))
