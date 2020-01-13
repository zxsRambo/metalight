class SOTLAgent():
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path):

        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.current_phase_time = 0

        self.DIC_PHASE_MAP = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            8: 7,
            0: 0
        }
        self.green_lane = {}
        start_lane = self.dic_traffic_env_conf["LANE_PHASE_INFO"]["start_lane"]
        for phase in self.dic_traffic_env_conf["LANE_PHASE_INFO"]["phase"]:
            self.green_lane[phase] = []
            for lane in self.dic_traffic_env_conf["LANE_PHASE_INFO"]["phase_startLane_mapping"][phase]:
                self.green_lane[phase].append(start_lane.index(lane))

        self.time_this_phase = self.dic_traffic_env_conf["MIN_ACTION_TIME"]
        self.phase_list = dic_traffic_env_conf["LANE_PHASE_INFO"]['phase']
        self.phase_id = 0


    def choose_action(self, state):
        ''' choose the best action for current state '''

        state = state[0][0]
        if state["cur_phase_index"][0] == -1:
            return self.phase_id

        cur_phase = self.DIC_PHASE_MAP[state["cur_phase_index"][0]]
        #print(state)


        if state["time_this_phase"][0] >= self.dic_agent_conf["PHI"] and cur_phase != -1:
            green_vec = sum([state["lane_num_vehicle_been_stopped_thres1"][i] for i in self.green_lane[cur_phase+1]])
            red_vec = sum(state["lane_num_vehicle_been_stopped_thres1"]) - green_vec
            print("green: %d, red: %d"%(green_vec, red_vec))
            if green_vec <= self.dic_agent_conf["MIN_GREEN_VEC"] and \
                red_vec > self.dic_agent_conf["MAX_RED_VEC"]:
                self.current_phase_time = 0
                self.action = (cur_phase + 1) % len(self.dic_traffic_env_conf["PHASE"])
                return (cur_phase + 1) % len(self.dic_traffic_env_conf["PHASE"])
            else:
                self.action = cur_phase
                self.current_phase_time += 1
                return cur_phase
        else:
            self.action = cur_phase
            self.current_phase_time += 1
            return cur_phase

