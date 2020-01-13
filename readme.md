# MetaLight
MetaLight is a value-based meta-reinforcement learning framework for traffic signal control. This repository includes the detailed implementation of our MetaLight [paper](https://uploads.strikinglycdn.com/files/f8450f43-19af-48ec-831a-bec87f1ec038/Metalight-AAAI-20.pdf) which was published in AAAI 2020.

Usage and more information can be found below.

## Dataset
This repo contrains four real-world datasets (Hangzhou, Jinan, Atlanta, and Los Angeles) which are stored in the `data.zip`. Please **unzip it** first before you run the code. More descriptions about the dataset can be found in our MetaLight paper.

## Installation Guide
### Dependencies
- python 3.*
- tensorflow v1.0+
- [cityflow](https://cityflow.readthedocs.io/en/latest/index.html)

### Quick Start 
We recommend to run the code through [docker](https://docs.docker.com/) and a sample docker image has been built for your quick start.

1. Please pull the docker image from the docker hub. 
``docker pull synthzxs/metalight:latest``
(Alternatively, you can use the *docker/dockerfile* to build the docker image directly on your linux system)
2. Please pull the codes for MetaLight from this repo.
``git clone https://github.com/zxsRambo/metalight.git``

3. Please run the built docker image, *synthzxs/metalight:latest*, to initiate a docker container. Please remember to mount the code directory.
``docker run -it -v /local_path_to_the_code/metalight/:/metalight/ synthzxs/metalight:latest /bin/bash``
Up to now, the experiment environment has been set up. Then, go the workspace in the docker contrainer, ``cd /metalight``, and try to run the code.



## Example 

Start an example:

``sh run_exp.sh``

In this utils, it runs the file ``meta_train.py``. Here are some important arguments that can be modified for different experiments:

* memo: the memo name of the experiment
* algorithm: the specified algorithm, e.g., MetaLight, FRAPPlus.

Hyperparameters such as learning rate, sample size and the like for the agent can also be assigned in our code and they are easy to tune for better performance.

## Link
- [FRAP](https://github.com/gjzheng93/frap-pub): base model for MetaLight
- [CityFlow](https://github.com/cityflow-project/CityFlow): traffic simulator
- [Collection of team works on RL for traffic signal control](https://github.com/traffic-signal-control/RL_signals)

## Citation
If you find this work useful in your research, please cite our [paper](https://uploads.strikinglycdn.com/files/f8450f43-19af-48ec-831a-bec87f1ec038/Metalight-AAAI-20.pdf) "MetaLight: Value-based Meta-reinforcement Learning for Traffic Signal Control". 
```
@inproceedings{metalight,
  title={MetaLight: Value-based Meta-reinforcement Learning for Traffic Signal Control},
  author={Zang, Xinshi and Yao, Huaxiu and Zheng, Guanjie and Xu, Nan and Xu, Kai and Li, Zhenhui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2020}
}
```
