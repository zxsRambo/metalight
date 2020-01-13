from traffic_meta_train import *
from traffic_meta_test import *


traffic_category = {
    "train": { # meta-train
        "4a": paper_meta_train_4a_phase_traffic_list,
        "4b": paper_meta_train_4b_phase_traffic_list,
        "6a": paper_meta_train_6a_phase_traffic_list,
        "6c": paper_meta_train_6c_phase_traffic_list,
        "6e": paper_meta_train_6e_phase_traffic_list,
        "8" : paper_meta_train_8_phase_traffic_list
    },
    "valid": { # meta-test task1
        "4a": paper_meta_valid_4a_phase_traffic_list,
        "4b": paper_meta_valid_4b_phase_traffic_list,
        "6a": paper_meta_valid_6a_phase_traffic_list,
        "6c": paper_meta_valid_6c_phase_traffic_list,
        "6e": paper_meta_valid_6e_phase_traffic_list,
        "8" : paper_meta_valid_8_phase_traffic_list
    },
    "test": { # meta-test task2
        "4c": paper_meta_test_4c_traffic_list,
        "4d": paper_meta_test_4d_traffic_list,
        "6b": paper_meta_test_6b_traffic_list,
        "6d": paper_meta_test_6d_traffic_list,
        "6f": paper_meta_test_6f_traffic_list
    },
    "city": city_train_phase, # manually change (meta-test task3 homo)
    #"city": city_test_phase, # manually change (meta-test task3 hete)
}

roadnet_map = {
    "4a": "roadnet_p4a_lt.json",
    "4b": "roadnet_p4b_lt.json",
    "4c": "roadnet_p4c_lt.json",
    "4d": "roadnet_p4d_lt.json",

    "6a": "roadnet_p6a_lt.json",
    "6b": "roadnet_p6b_lt.json",
    "6c": "roadnet_p6c_lt.json",
    "6d": "roadnet_p6d_lt.json",
    "6e": "roadnet_p6e_lt.json",
    "6f": "roadnet_p6f_lt.json",

    "8": "roadnet_p8_lt.json",
}

flow_map = {
    "4": "flow.json",
    "3e": "flow_3e.json",
    "3n": "flow_3n.json",
    "3s": "flow_3s.json",
    "3w": "flow_3w.json"
}

meta_train_traffic = [t for type in traffic_category["train"] for t in traffic_category["train"][type]]
meta_valid_traffic = [t for type in traffic_category["valid"] for t in traffic_category["valid"][type]]
meta_test_traffic = [t for type in traffic_category["test"] for t in traffic_category["test"][type]]
meta_test_city = [t for type in traffic_category["city"] for t in traffic_category["city"][type]]

traffic_category["train_all"] = meta_train_traffic
traffic_category["valid_all"] = meta_valid_traffic
traffic_category["test_all"] = meta_test_traffic
traffic_category["city_all"] = meta_test_city

traffic_category["traffic_info"] = {}
for ctg in ["train", "valid", "test", "city"]:
    for type in traffic_category[ctg].keys():
        if "3" in roadnet_map[type]:
            flow_file = flow_map[roadnet_map[type][5:7]]
        else:
            flow_file = flow_map["4"]
        for traffic in traffic_category[ctg][type]:
            if traffic in traffic_category["traffic_info"].keys():
                print("old: ", traffic_category["traffic_info"][traffic])
                print("new: ", (ctg, type, roadnet_map[type], flow_file))
                print("traffic info is already set! Attention duplicate!")
                raise(ValueError)
            traffic_category["traffic_info"][traffic] = (ctg, type, roadnet_map[type], flow_file)
