import tensorflow as tf
import utils
import numpy as np

from agent import Agent

def relation(lane_phase_info):
    relations = []
    num_phase = len(lane_phase_info["phase"])
    map = lane_phase_info["phase_roadLink_mapping"]
    for p1 in lane_phase_info["phase"]:
        zeros = [0] * (num_phase - 1)
        count = 0
        for p2 in lane_phase_info["phase"]:
            if p1 == p2:
                continue
            if len(set(map[p1] + map[p2])) != len(map[p1]) + len(map[p2]):
                zeros[count] = 1
            count += 1
        relations.append(zeros)
    relations = np.array(relations).reshape(1, num_phase, num_phase - 1)
    constant = relations
    return constant

class FRAPPlusAgent(Agent):
    """
        FRAP++ makes a few improvements to FRAP (https://github.com/gjzheng93/frap-pub) and is also a Tensorflow version
    """
    def construct_weights(self, dim_input, dim_output):
        weights = {}

        weights['embed_w1'] = tf.Variable(tf.glorot_uniform_initializer()([1, 4]), name='embed_w1')
        weights['embed_b1'] = tf.Variable(tf.zeros([4]), name='embed_b1')

        # for phase, one-hot
        weights['embed_w2'] = tf.Variable(tf.random_uniform_initializer(minval=-0.05, maxval=0.05)([2, 4]), name='embed_w2')

        # lane embeding
        weights['lane_embed_w3'] = tf.Variable(tf.glorot_uniform_initializer()([8, 16]), name='lane_embed_w3')
        weights['lane_embed_b3'] = tf.Variable(tf.zeros([16]), name='lane_embed_b3')

        # relation embeding, one-hot
        weights['relation_embed_w4'] = tf.Variable(tf.random_uniform_initializer(minval=-0.05, maxval=0.05)([2, 4]), name='relation_embed_w4')

        weights['feature_conv_w1'] = tf.Variable(tf.glorot_uniform_initializer()([1, 1, 32, self.dic_agent_conf["D_DENSE"]]), name='feature_conv_w1')
        weights['feature_conv_b1'] = tf.Variable(tf.zeros([self.dic_agent_conf['D_DENSE']]), name='feature_conv_b1')

        weights['phase_conv_w1'] = tf.Variable(
            tf.glorot_uniform_initializer()([1, 1, 4, self.dic_agent_conf["D_DENSE"]]), name='phase_conv_w1')
        weights['phase_conv_b1'] = tf.Variable(tf.zeros([self.dic_agent_conf['D_DENSE']]), name='phase_conv_b1')

        weights['combine_conv_w1'] = tf.Variable(
            tf.glorot_uniform_initializer()([1, 1, self.dic_agent_conf["D_DENSE"], self.dic_agent_conf["D_DENSE"]]), name='combine_conv_w1')
        weights['combine_conv_b1'] = tf.Variable(tf.zeros([self.dic_agent_conf['D_DENSE']]), name='combine_conv_b1')

        weights['final_conv_w1'] = tf.Variable(
            tf.glorot_uniform_initializer()([1, 1, self.dic_agent_conf["D_DENSE"], 1]), name='final_conv_w1')
        weights['final_conv_b1'] = tf.Variable(tf.zeros([1]), name='final_conv_b1')

        return weights

    def construct_forward(self, inp, weights, reuse, norm, is_train, prefix='fc'):
        dim = int(inp.shape[1].value / 2)
        num_veh = inp[:, :dim]
        batch_size = num_veh.shape[0]
        num_veh = tf.reshape(num_veh, [-1, 1])

        phase = inp[:, dim:]
        phase = tf.cast(phase, tf.int32)
        phase = tf.one_hot(phase, 2)
        phase = tf.reshape(phase, [-1, 2])

        embed_num_veh = utils.contruct_layer(tf.matmul(num_veh, weights['embed_w1']) + weights['embed_b1'],
                                 activation_fn=tf.nn.sigmoid, reuse=reuse, is_train=is_train,
                                 norm=norm, scope='num_veh_embed.' + prefix
                                 )
        embed_num_veh = tf.reshape(embed_num_veh, [-1, dim, 4])

        embed_phase = utils.contruct_layer(tf.matmul(phase, weights['embed_w2']),
                                 activation_fn=tf.nn.sigmoid, reuse=reuse, is_train=is_train,
                                 norm=norm, scope='phase_embed.' + prefix
                                 )
        embed_phase = tf.reshape(embed_phase, [-1, dim, 4])

        dic_lane = {}
        for i, m in enumerate(self.dic_traffic_env_conf["LANE_PHASE_INFO"]["start_lane"]):
            dic_lane[m] = tf.concat([embed_num_veh[:, i, :], embed_phase[:, i, :]], axis=-1)


        list_phase_pressure = []
        phase_startLane_mapping = self.dic_traffic_env_conf["LANE_PHASE_INFO"]["phase_startLane_mapping"]
        for phase in self.dic_traffic_env_conf["LANE_PHASE_INFO"]["phase"]:
            t1 = tf.Variable(tf.zeros(1))
            t2 = tf.Variable(tf.zeros(1))
            # FRAP ++: change "Sum" to "Average"
            #for lane in phase_startLane_mapping[phase][0]:
            #    t1 += utils.contruct_layer(
            #       tf.matmul(dic_lane[lane], weights['lane_embed_w3']) + weights['lane_embed_b3'],
            #       activation_fn=self._activation_fn, reuse=reuse, is_train=is_train,
            #       norm=norm, scope='lane_embed.' + prefix
            #       )
            #t1 /= len(phase_startLane_mapping[phase][0])
            #if len(phase_startLane_mapping[phase]) >= 2:
            #    for lane in phase_startLane_mapping[phase][1]:
            #        t2 += utils.contruct_layer(
            #           tf.matmul(dic_lane[lane], weights['lane_embed_w3']) + weights['lane_embed_b3'],
            #           activation_fn=self._activation_fn, reuse=reuse, is_train=is_train,
            #           norm=norm, scope='lane_embed.' + prefix
            #           )
            #    t2 /= len(phase_startLane_mapping[phase][1])
            #list_phase_pressure.append(t1 + t2)

            for lane in phase_startLane_mapping[phase]:
                t1 += utils.contruct_layer(
                   tf.matmul(dic_lane[lane], weights['lane_embed_w3']) + weights['lane_embed_b3'],
                   activation_fn=self._activation_fn, reuse=reuse, is_train=is_train,
                   norm=norm, scope='lane_embed.' + prefix
                   )
            t1 /= len(phase_startLane_mapping[phase])
            list_phase_pressure.append(t1)


        constant = relation(self.dic_traffic_env_conf["LANE_PHASE_INFO"])

        constant = tf.one_hot(constant, 2)
        s1, s2 = constant.shape[1:3]
        constant = tf.reshape(constant, (-1, 2))
        relation_embedding = tf.matmul(constant, weights['relation_embed_w4'])
        relation_embedding = tf.reshape(relation_embedding, (-1, s1, s2, 4))

        list_phase_pressure_recomb = []
        num_phase = len(list_phase_pressure)

        for i in range(num_phase):
            for j in range(num_phase):
                if i != j:
                    list_phase_pressure_recomb.append(
                        tf.concat([list_phase_pressure[i], list_phase_pressure[j]], axis=-1,
                                    name="concat_compete_phase_%d_%d" % (i, j)))

        list_phase_pressure_recomb = tf.concat(list_phase_pressure_recomb, axis=-1 , name="concat_all")
        feature_map = tf.reshape(list_phase_pressure_recomb, (-1, num_phase, num_phase-1, 32))

        lane_conv = tf.nn.conv2d(feature_map, weights['feature_conv_w1'], [1, 1, 1, 1], 'VALID', name='feature_conv') + weights['feature_conv_b1']
        lane_conv = tf.nn.leaky_relu(lane_conv, name='feature_activation')

        # relation conv layer
        relation_conv = tf.nn.conv2d(relation_embedding, weights['phase_conv_w1'], [1, 1, 1, 1], 'VALID',
                                 name='phase_conv') + weights['phase_conv_b1']
        relation_conv = tf.nn.leaky_relu(relation_conv, name='phase_activation')
        combine_feature = tf.multiply(lane_conv, relation_conv, name="combine_feature")

        # second conv layer
        hidden_layer = tf.nn.conv2d(combine_feature, weights['combine_conv_w1'], [1, 1, 1, 1], 'VALID', name='combine_conv') + \
                    weights['combine_conv_b1']
        hidden_layer = tf.nn.leaky_relu(hidden_layer, name='combine_activation')

        before_merge = tf.nn.conv2d(hidden_layer, weights['final_conv_w1'], [1, 1, 1, 1], 'VALID',
                                    name='final_conv') + \
                       weights['final_conv_b1']

        _shape = (-1, self.num_actions, self.num_actions-1)
        before_merge = tf.reshape(before_merge, _shape)
        out = tf.reduce_sum(before_merge, axis=2)

        return out

    def init_params(self):
        return self.save_params()
