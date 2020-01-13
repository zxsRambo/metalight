import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils
import config
import time

class Agent():
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path):
        t1 = time.time()
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self._is_train = True
        self._alpha = self.dic_agent_conf['ALPHA']
        self._min_alpha = self.dic_agent_conf['MIN_ALPHA']
        self._alpha_decay_rate = self.dic_agent_conf['ALPHA_DECAY_RATE']
        self._alpha_decay_step = self.dic_agent_conf['ALPHA_DECAY_STEP']
        self._K = 1
        self._norm = self.dic_agent_conf['NORM']#'None' #'batch_norm'
        self._batch_size = 20
        self._num_updates = self.dic_agent_conf['NUM_UPDATES']
        self._avoid_second_derivative = False

        self._loss_fn = self._get_loss_fn('MSE')

        if self.dic_agent_conf['ACTIVATION_FN'] == 'relu':
            self._activation_fn = tf.nn.relu
        elif self.dic_agent_conf['ACTIVATION_FN'] == 'leaky_relu':
            self._activation_fn = tf.nn.leaky_relu
        else:
            raise(ValueError)

        ## dimension of input and output
        if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
            self.num_actions = 2
        else:
            self.num_actions = dic_traffic_env_conf["num_phases"]
        self.num_phases = dic_traffic_env_conf["num_phases"]
        self.num_lanes = dic_traffic_env_conf["num_lanes"]

        ## others
        if self.num_lanes == 1:
            self.dic_phase_expansion = config.dic_two_phase_expansion
        elif self.num_lanes == 2:
            self.dic_phase_expansion = config.dic_four_phase_expansion

        self.dim_input = 0
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if "phase" in feature_name and self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                self.dim_input += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()][0]*self.num_lanes*4
            elif "phase" in feature_name and not self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                self.dim_input += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]
            else:
                self.dim_input += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]*self.num_lanes


        self._weights = self.construct_weights(self.dim_input, self.num_actions)
        self._build_placeholder()
        self._build_graph(self.dim_input, self.num_actions, norm=self._norm)
        self._assign_op = [self._weights[key].assign(self._weights_inp[key]) for key in self._weights.keys()]
        self._meta_grads = dict(zip(self._weights.keys(), tf.gradients(self._meta_loss, list(self._weights.values()))))

        self._sess = utils.get_session(1)
        self._sess.run(tf.global_variables_initializer())
        print("build policy time:", time.time() - t1)

    def _build_graph(self, dim_input, dim_output, norm):
        def model_summary():
            model_vars = tf.trainable_variables()
            slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        learning_x, learning_y, meta_x, meta_y = [self._learning_x, self._learning_y,
                                  self._meta_x, self._meta_y]
        learning_loss_list = []
        meta_loss_list = []

        weights = self._weights
        learning_output = self.construct_forward(learning_x, weights,
                                                   reuse=False, norm=norm,
                                                   is_train=self._is_train)

        # Meta train loss: Calculate gradient
        learning_loss = self._loss_fn(learning_y, learning_output)
        learning_loss = tf.reduce_mean(learning_loss)
        learning_loss_list.append(learning_loss)
        grads = dict(zip(weights.keys(),
                         tf.gradients(learning_loss, list(weights.values()))))
        # learning rate
        self.learning_rate_op = tf.maximum(self._min_alpha,
                                           tf.train.exponential_decay(
                                               self._alpha,
                                               self.alpha_step,
                                               self._alpha_decay_step,
                                               self._alpha_decay_rate,
                                               staircase=True
                                           ))
        self.learning_train_op = tf.train.AdamOptimizer(self.learning_rate_op).minimize(learning_loss)
        if self.dic_agent_conf['GRADIENT_CLIP']:
            for key in grads.keys():
                grads[key] = tf.clip_by_value(grads[key], -1 * self.dic_agent_conf['CLIP_SIZE'], self.dic_agent_conf['CLIP_SIZE'])

        self._learning_grads = grads
        new_weights = dict(zip(weights.keys(), [weights[key] - self.learning_rate_op * grads[key]
                                for key in weights.keys()]))

        if self._avoid_second_derivative:
            new_weights = tf.stop_gradients(new_weights)
        meta_output = self.construct_forward(meta_x, new_weights,
                                                 reuse=True, norm=norm,
                                                 is_train=self._is_train)
        # Meta val loss: Calculate loss (meta step)
        meta_loss = self._loss_fn(meta_y, meta_output)
        meta_loss = tf.reduce_mean(meta_loss)
        meta_loss_list.append(meta_loss)
        # If perform multiple updates

        for _ in range(self._num_updates - 1):
            learning_output = self.construct_forward(learning_x, new_weights,
                                                       reuse=True, norm=norm,
                                                       is_train=self._is_train)
            learning_loss = self._loss_fn(learning_y, learning_output)
            learning_loss = tf.reduce_mean(learning_loss)
            learning_loss_list.append(learning_loss)
            grads = dict(zip(new_weights.keys(),
                             tf.gradients(learning_loss, list(new_weights.values()))))
            new_weights = dict(zip(new_weights.keys(),
                                   [new_weights[key] - self.learning_rate_op * grads[key]
                                    for key in new_weights.keys()]))
            if self._avoid_second_derivative:
                new_weights = tf.stop_gradients(new_weights)
            meta_output = self.construct_forward(meta_x, new_weights,
                                                     reuse=True, norm=norm,
                                                     is_train=self._is_train)
            meta_loss = self._loss_fn(meta_y, meta_output)
            meta_loss = tf.reduce_mean(meta_loss)
            meta_loss_list.append(meta_loss)

        self._new_weights = new_weights

        # output
        self._learning_output = learning_output
        self._meta_output = meta_output

        # Loss
        learning_loss = tf.reduce_mean(learning_loss_list[-1])
        meta_loss = tf.reduce_mean(meta_loss_list[-1])

        self._learning_loss = learning_loss
        self._meta_loss = meta_loss
        model_summary()

    def _get_loss_fn(self, loss_type):
        if loss_type == 'MSE':
            loss_fn = tf.losses.mean_squared_error
        else:
            ValueError("Can't recognize the loss type {}".format(loss_type))
        return loss_fn

    def learning_predict(self, learning_x):
        with self._sess.as_default():
            with self._sess.graph.as_default():
                feed_dict = {
                    self._learning_x: learning_x
                }
                return self._sess.run(self._learning_output, feed_dict=feed_dict)

    def meta_predict(self, meta_x):
        with self._sess.as_default():
            with self._sess.graph.as_default():
                feed_dict = {
                    self._meta_x: meta_x
                }
                return self._sess.run(self._meta_output, feed_dict=feed_dict)

    def _build_placeholder(self):
        self.alpha_step = tf.placeholder('int64', None, name='alpha_step')
        self._learning_x = tf.placeholder(tf.float32, shape=(None, self.dim_input))
        self._learning_y = tf.placeholder(tf.float32, shape=(None, self.num_actions))
        self._meta_x = tf.placeholder(tf.float32, shape=(None, self.dim_input))
        self._meta_y = tf.placeholder(tf.float32, shape=(None, self.num_actions))
        self._weights_inp = {}
        for key in self._weights.keys():
            self._weights_inp[key] = tf.placeholder(tf.float32, shape=self._weights[key].shape)

    def choose_action(self, state, test=False):
        ''' choose the best action for current state '''
        inputs = [[] for _ in state]

        all_start_lane = self.dic_traffic_env_conf["LANE_PHASE_INFO"]["start_lane"]
        for i in range(len(state)):
            s = state[i]
            s = s[0]  ## Todo care about support multi_intersection
            inputs[i].extend(s['lane_num_vehicle'] + s["cur_phase"])
        inputs = np.reshape(np.array(inputs), (len(inputs), -1))
        q_values = self.learning_predict(inputs)

        if not test:
            if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
                action = np.array([random.randrange(q_values.shape[1]) for _ in range(q_values.shape[0])])
            else:  # exploitation
                action = np.argmax(q_values, axis=1)  # q_values shape: (2, 1, 8)
        else:
            action = np.argmax(q_values, axis=1)

        return action

    def decay_epsilon(self, batch_id):
        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], batch_id)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def fit(self, episodes, params, target_params):
        self.load_params(params)
        input_x = episodes.get_x()
        q_values = self.learning_predict(input_x)

        self.load_params(target_params)
        input_next_x = episodes.get_next_x()
        target_q_values = self.learning_predict(input_next_x)

        for i in range(len(episodes.total_samples)):
            sample = episodes.total_samples[i]
            action = sample[1][0]
            reward = sample[3][0]
            q_values[i][action] = reward + self.dic_agent_conf['GAMMA'] * np.max(target_q_values[i])

        episodes.prepare_y(q_values)

    def update_params(self, episodes, params, lr_step, slice_index):
        learning_x = episodes.get_x()[slice_index]
        learning_y = episodes.get_y()[slice_index]
        print("Task | Traffic:", self.dic_traffic_env_conf['TRAFFIC_FILE'])
        t1 = time.time()

        if self.dic_agent_conf['OPTIMIZER'] == 'sgd':
            for i in range(self.dic_agent_conf['NUM_GRADIENT_STEP']):
                self.load_params(params)
                with self._sess.as_default():
                    with self._sess.graph.as_default():
                        feed_dict = {
                            self._learning_x: learning_x,
                            self._learning_y: learning_y,
                            self.alpha_step: lr_step
                        }
                        params, learning_loss, lr = self._sess.run([self._new_weights, self._learning_loss, self.learning_rate_op], feed_dict=feed_dict)
                        print("step: %d, epoch: %3d, loss: %f, learning_rate: %f, epsilon: %f" % (
                            lr_step, i, learning_loss, lr, self.dic_agent_conf["EPSILON"]))
        elif self.dic_agent_conf['OPTIMIZER'] == 'adam':
            _weights_list = list(self._weights.values())

            for i in range(self.dic_agent_conf['NUM_GRADIENT_STEP']):
                with self._sess.as_default():
                    with self._sess.graph.as_default():
                        feed_dict = {
                            self._learning_x: learning_x,
                            self._learning_y: learning_y,
                            self.alpha_step: lr_step
                        }
                        _, weights_list, learning_loss, lr = self._sess.run([self.learning_train_op, _weights_list, self._learning_loss, self.learning_rate_op], feed_dict=feed_dict)
                        print("step: %d, epoch: %3d, loss: %f, learning_rate: %f, epsilon: %f" % (
                            lr_step, i, learning_loss, lr, self.dic_agent_conf["EPSILON"]))
            params = dict(zip(self._weights.keys(), weights_list))
        else:
            raise(NotImplementedError)
        t2 = time.time()
        return params

    def load_params(self, params):
        with self._sess.as_default():
           with self._sess.graph.as_default():
               feed_dict = {self._weights_inp[key]: params[key] for key in self._weights.keys()}
               self._sess.run(self._assign_op, feed_dict=feed_dict)

    def save_params(self):
        with self._sess.as_default():
            with self._sess.graph.as_default():
                return self._sess.run(self._weights)

    def cal_grads(self, learning_episodes, meta_episodes, slice_index, params):
        self.load_params(params)
        t1 = time.time()

        if not second_index:
            second_index = slice_index

        with self._sess.as_default():
            with self._sess.graph.as_default():
                feed_dict = {
                    self._learning_x: learning_episodes.get_x()[slice_index],
                    self._learning_y: learning_episodes.get_y()[slice_index],
                    self._meta_x: meta_episodes.get_x()[second_index],
                    self._meta_y: meta_episodes.get_y()[second_index],
                    self.alpha_step: 0, # TODO hard code
                }
                res = self._sess.run(self._meta_grads, feed_dict=feed_dict)
        t2 = time.time()
        return res

    def second_cal_grads(self, episodes, slice_index, new_slice_index, params):
        self.load_params(params)
        t1 = time.time()

        with self._sess.as_default():
            with self._sess.graph.as_default():
                feed_dict = {
                    self._learning_x: episodes.get_x()[slice_index],
                    self._learning_y: episodes.get_y()[slice_index],
                    self._meta_x: episodes.get_x()[new_slice_index],
                    self._meta_y: episodes.get_y()[new_slice_index],
                    self.alpha_step: 0,  # TODO hard code
                }
                res = self._sess.run(self._meta_grads, feed_dict=feed_dict)
        t2 = time.time()
        return res