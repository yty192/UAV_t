import random
import scipy.special
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer, Flatten, Conv2D, Conv1D, Activation, LeakyReLU
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import Adam
from matplotlib import pylab as plt
import scipy

#import threading
#import time

import math
random_Seed = 100
import tensorflow as tf
from keras import initializers
from keras import regularizers

class Users:
    def __init__(self) :
        usernum = 100

    def initial_coordinate(self):
        ran = ((2*r_c)**2)*MTs_density/(10**6)
        print(ran)
        sum_package = 0
        for i in range(0, int(ran)):
            x = random.uniform(-r_c, r_c)
            y = random.uniform(-r_c, r_c)
            if (x**2 + y**2 > r_t**2) and (x**2 + y**2 <= r_c**2):
                package = random.uniform(p_min, p_max)
                sum_package += package
                list_in.append([i, x, y, package])      #initiae users in Ring
                #print('yes')
        return list_in, sum_package


class Environment:
    def __init__(self):
        #self.MTs_density = 0.0001        # density: lambda  MTs/m^2
        self.r_c = 1000              # r_c:cell_radius, total number: K=pai*(r_G^2)*lambda, KK={1,2,...,K}
        self.r_t = 500             # r_t:distance_threshold, to the GBS

        self.taskCount = 0            # number of total task
        self.successTaskCount = 0    # number of successfully offloaded tasks

        self.success_bit = 0
        self.select_bit = 0

        self.H_U = 100  # H_U:fly_altitude
        self.V = 15  # V:UAV_speed m/s
        self.r_u = 740  # UAV follows a circle trajectory r_u:trajectory_radius
        self.r_cover = 300
        # flying duration:  T=2*pai*(r_U)/V
        self.theta_U = (math.atan(self.r_cover/self.H_U)/(math.pi*2))*360 # angle of UAV theta_U:antenna_angle
        # r_cover = H_U*tan(theta_U)~=300

    def reset(self):
        print('env reset')
        self.taskCount = 0
        self.successTaskCount = 0
        self.success_bit = 0
        self.select_bit = 0

    def list_update(self, j1, total_size1, list_in2):
        list_now1 = []
        time_list = j1*tc       # acquire a time compared to time:0
        print('time now:', time_list)
        theta_UAV = (time_list-t0)*V/self.r_u    # unit is rad  0~2pi
        #print('theta is:', theta_UAV)
        X0 = self.r_u * math.cos(theta_UAV)       # X0, Y0:  UAV
        Y0 = self.r_u * math.sin(theta_UAV)
        #print('X0 is:', X0, 'Y0 is:', Y0)
        for i in range(total_size1):
            distance = math.sqrt((list_in2[i][1]-X0)**2 + (list_in2[i][2]-Y0)**2)
            C = B*math.log2(1+P*A/(N*(distance**2)))
            taskDelay = list_in2[i][3]/C
            if distance < self.r_cover:
                theta_user = math.atan2(list_in2[i][2], list_in2[i][1])   # unit is rad
                if list_in2[i][2] < 0:
                    theta_user += 2*math.pi
                if theta_user > theta_UAV:
                    distance = distance
                else:
                    distance = -distance

                list_now1.append([list_in2[i][0], list_in2[i][1], list_in2[i][2], distance, list_in2[i][3], taskDelay, C])    #  7 kinds of data

        return list_now1

    def judgement(self, list_now2, process_number2, package_task2):# When a submission is accomplished, judge whether this user is still
        task_success1 = False
        for j in range(len(list_now2)):  # in the coverage.
            if list_now2[j][0] == process_number2:
                self.successTaskCount += 1
                self.success_bit += package_task2
                task_success1 = True
        self.taskCount += 1
        self.select_bit += package_task2
        print('success task in judgement:', self.successTaskCount, 'total task in judgement:', self.taskCount)
        ratio1 = self.successTaskCount/self.taskCount
        return task_success1, ratio1

    def judge(self, list_now3, process_number1, package_task1):    # When the submission is not finished, judge whether this user is still
        in_list = 0                # in the coverage.
        not_in_list = 0
        aa = 0
        for k in range(len(list_now3)):
            if list_now3[k][0] == process_number1:
                in_list = 1
                aa = k
            else:
                not_in_list = 1
        if in_list == 0 and not_in_list == 1:
            self.taskCount += 1
            self.select_bit += package_task1
            circle_out1 = True
        else:
            circle_out1 = False
        return circle_out1, aa

    def ratio_result(self, sum_package2):
        ratio2 = self.successTaskCount / self.taskCount
        ratio3 = self.successTaskCount/total_size
        ratio_bit = self.success_bit/self.select_bit
        ratio_bit_total = self.success_bit/sum_package2
        return ratio2, ratio3, ratio_bit, ratio_bit_total

    def getState(self, list_now10, action_size10):    ########### no serial number
        if action_size10 <= density_value:
            # state = [[ss[3], ss[4]] for ss in list_now]
            state = np.zeros((1, density_value, 2))
            #state[0, 0:action_size10, 0] = ([ss[0] for ss in list_now10])  ## serial number
            state[0, 0:action_size10, 0] = ([ss[3]/(math.sqrt(self.H_U**2+self.r_cover**2)) for ss in list_now10])#d
            state[0, 0:action_size10, 1] = ([(ss[4]-p_min)/(p_max-p_min) for ss in list_now10])  ## package
        else:
            while action_size10 - density_value > 0:
                del list_now10[random.randint(0, (action_size10 - 1))]  ## two ends
                action_size10 -= 1
            state = np.zeros((1, density_value, 2))
            #state[0, 0:action_size10, 0] = ([ss[0] for ss in list_now10])  ## serial number
            state[0, 0:action_size10, 0] = ([ss[3]/(math.sqrt(self.H_U**2+self.r_cover**2)) for ss in list_now10])
            state[0, 0:action_size10, 1] = ([(ss[4]-p_min)/(p_max-p_min) for ss in list_now10]) ## package
        return state, list_now10, action_size10


class Agent:
    def __init__(self, action_size1, network_model):
        self.state_size = density_value*2     #############
        self.action_size = action_size1
        self.memory_size = 10000
        self.prioritized = True
        #print('action size is:', action_size1)
        if self.prioritized:
            self.memory = Memory(capacity=self.memory_size)
        # else:
        #     self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        #     #self.memory = deque(maxlen=self.memory_size)
        self.gamma = 0.9      # decay rate
        self.epsilon = 1.0     # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9993
        self.learning_rate = 0.001

        self.loss_record = []
        self.number_reply_record = []
        self.Q_value_record = []
        self.test_Q_value_record = []
        self.number_Q_record = []
        self.test_number_Q_record = []
        self.exploration = True
        if network_model == 0:
            self.model = self._build_NN_model()
            self.model_target = self.model     ####################
        if network_model == 1:
            self.model = load_model(filepath="model.h5")
            self.model_target = self.model     #######################
            self.exploration = False
        if network_model == 2:
            self.model = self._build_Con_model()
            self.model_target = self.model
        self.Q_value = 0
        self.test_Q_value = 0
        self.number_Q = 0
        self.test_number_Q =0
        self.number_reply = 0
        self.loss = 0
        self.alpha = 0.1

    def reset(self):
        print('agent reset')
        self.Q_value = 0

    def _build_NN_model(self):
        model = Sequential()
        #model.add(Flatten(input_shape=(10, 3)))     ###### state......
        model.add(InputLayer(input_shape=(density_value, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation = 'relu', kernel_initializer="zeros", bias_initializer="zeros"))
        #model.add(Activation('relu'))
        #model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(128,activation = 'relu',kernel_initializer="zeros",bias_initializer="zeros"))
        #model.add(Activation('relu'))
        #model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(128, activation = 'relu',kernel_initializer="zeros", bias_initializer="zeros"))
        #model.add(Dense(128, kernel_initializer="zeros",bias_initializer="zeros",kernel_regularizer=regularizers.l2(0.01)))
        #model.add(Activation('relu'))
        #model.add(LeakyReLU(alpha=0.1))
        #model.add(Flatten())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_Con_model(self):
        # 1D Convolutional Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv1D(20, 2, input_shape=(density_value, 2), kernel_initializer="zeros", bias_initializer="zeros"))
        model.add(Flatten())
        model.add(Dense(512, activation='relu',kernel_initializer="zeros", bias_initializer="zeros"))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # def remember(self, state, action, reward, next_state):
    #     self.memory.append((state, action, reward, next_state))

    def store_transition(self, state, action, reward, next_state):
        state_ = state.reshape(1*density_value*2)
        next_state_ = next_state.reshape(1 * density_value * 2)
        transition = np.hstack((state_, [action, reward], next_state_))
        self.memory.store(transition)  # have high priority for newly arrived transition

    def act(self, list_now4):   ### list_now4 is state
        list_distance = []
        list_package = []
        list_distance_n = []
        if method == 0:
            return random.randrange(self.action_size)     #epsilon=1, it is random now
        elif method ==1:
            list_distance = [math.fabs(d[3]) for d in list_now4]
            index_distance = list_distance.index(min(list_distance))     ## +distance min
            return index_distance
        elif method == 2:
            list_package = [p[4] for p in list_now4]
            index_package = list_package.index(min(list_package))
            return index_package
        elif method ==3:
            list_distance_n = [dn[3] for dn in list_now4]
            index_distance_n = list_distance_n.index(min(list_distance_n))    ## -distance min
            return index_distance_n
        elif method ==4:
            list_distance_p = [dp[3] for dp in list_now4]
            index_distance_p = list_distance_p.index(max(list_distance_p))    ## distance max
            return index_distance_p
        elif method == 10:
            #print('!!!!!!!!!!!!!!!!!!!!!!!!', self.exploration)
            if (np.random.rand() <= self.epsilon) and self.exploration:
                actionIndex = random.randrange(self.action_size)
                while list_now4[0][actionIndex][1] == 0:
                    actionIndex = random.randrange(self.action_size)
                # if self.epsilon < 0.2:
                #     while list_now4[0][actionIndex][1] == 0:
                #         actionIndex = random.randrange(self.action_size)

            else:
                act_values = self.model.predict(state)      ###########################
                if self.exploration is False:
                    act_values = self.model_target.predict(state)
                print('@@@@@@@@@@@ act_values is ', act_values)

                actionIndex = np.argmax(act_values[0])
                # if self.epsilon < 0.2:
                #     act_value = self.model_target.predict(state)
                #     self.Q_value = np.max(act_value)
                #
                if self.exploration:
                    self.number_Q += 1
                    self.number_Q_record.append(self.number_Q)
                    list_all_q.append(act_values[0])
                    self.Q_value = np.max(act_values[0])
                    self.Q_value_record.append(self.Q_value)

                else:
                    self.test_number_Q += 1
                    self.test_number_Q_record.append(self.test_number_Q)
                    self.test_Q_value = np.max(act_values[0])
                    self.test_Q_value_record.append(self.test_Q_value)

            if self.exploration:
                if (self.epsilon > self.epsilon_min) and self.exploration:
                    self.epsilon *= self.epsilon_decay

            return actionIndex

        else:
            print('method cannot find')

    def replay(self, batch_size):
        if (self.number_reply % 50) and (self.number_reply != 0) == 0:
            self.model_target = self.model  # target network update every 50 trainings
        tree_idx, batch_memory, ISWeights = self.memory.sample(batch_size)
        index = np.arange(batch_size, dtype=np.int32)

        statess = batch_memory[:, :self.state_size]
        next_statess = batch_memory[:, -self.state_size:]
        states = statess.reshape(batch_size, density_value, 2)
        next_states = next_statess.reshape(batch_size, density_value, 2)

        targets = np.empty((batch_size, self.action_size))
        targets_f = np.empty((batch_size, self.action_size))
        for r in range(batch_size):
            action = batch_memory[r, self.state_size].astype(int)
            reward = batch_memory[r, self.state_size + 1]
            next_states_ = next_states[r].reshape(1,density_value, 2)
            target = reward + self.gamma * np.max(self.model_target.predict(next_states_)[0])
            states_ = states[r].reshape(1, density_value, 2)
            target_f = self.model.predict(states_)
            targets_f[r, :] = target_f
            target_f[0][action] = target  #######################################################
            targets[r, :] = target_f

        self.number_reply += 1
        self.number_reply_record.append(self.number_reply)
        self.loss = self.model.train_on_batch(states, targets)

        abs_errors = tf.reduce_sum(tf.abs(targets - targets_f), axis=1)
        print('abs_errors is', abs_errors)
        self.loss = tf.reduce_mean(ISWeights * tf.squared_difference(targets, targets_f))
        self.loss_record.append(self.loss)
        self.memory.batch_update(tree_idx, abs_errors)


    def Q_value_compute(self):
        return self.Q_value_record, self.number_Q_record

    def test_Q_value_compute(self):
        return self.test_Q_value_record, self.test_number_Q_record

    def loss_compute(self):
        return self.loss_record, self.number_reply_record

    def save(self):
        self.model.save(filepath="model.h5")
        #self.model.load_weights('model.h5')

    def longterm_reward_compute(self, reward0):     # long_term reward
        reward_memory.append((reward0))
        reward1 = np.mean(reward_memory)
        return reward1


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    # store data, update sumtree
    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    # get sample
    def sample(self, n):    # n is batch_size
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for _i in range(n):
            _a, b = pri_seg * _i, pri_seg * (_i + 1)
            v = np.random.uniform(_a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[_i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[_i], b_memory[_i, :] = idx, data
        return b_idx, b_memory, ISWeights

    # after training, update sample's priority in the tree
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class SumTree(object):
    data_pointer = 0

    # build tree and data
    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2*capacity - 1)  # for all parent nodes and leaves
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    # when there is a new sample, add in tree and data
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data frame
        self.update(tree_idx, p)  # update tree frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0    # replace when exceed the capacity

    # when samples are trained, new TD errors are generated, update tree
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    # get samples according to selected v
    def get_leaf(self, v):
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


def test_for_real_success_total_bit(agent):
    test_test_flag = False
    random.seed(random_Seed)
    test_env = Environment()
    test_user = Users()
    for u in range(1):
        #random.seed(11)   #just 1 round,same in or out
        test_env.reset()
        test_Ts = (2 * math.pi * r_u) / V  # time of flying a circle by UAV
        test_tc = 0.5  # decision time
        test_counter = math.ceil(test_Ts / test_tc)  # one circle has number of counter operations

        test_list_in = []
        test_list_now = []


        test_package_task = 0
        test_time_task = 0
        test_capacity_task = 0
        test_task_remain = 0
        test_c = 0

        test_success_bit = 0
        test_total_bit = 0

        test_list_in1, test_sum_package1 = test_user.initial_coordinate()
        test_total_size = len(test_list_in1)

        test_t0 = 0  # not used, use t0 actually

        for v in range(test_counter):
            test_list_now0 = test_env.list_update(v, test_total_size,test_list_in1)

            test_action_size0 = len(test_list_now0)
            if test_action_size0 == 0:
                continue
            test_state, test_list_now, test_action_size = test_env.getState(test_list_now0, test_action_size0)

            if v == 0 or test_test_flag is True:
                test_a = agent.act(test_state)
                list_action.append([test_a])

                if test_a >= len(test_list_now):  ### 123456 are all ok
                    test_state_old = test_state
                    test_test_flag = True
                    continue

                test_test_flag = False
                test_state_old = test_state
                test_process_number = test_list_now[test_a][0]  # serial number
                test_package_task = test_list_now[test_a][4]
                test_task_remain = test_package_task  # after act, task_remain equals package_task
                test_time_task = test_list_now[test_a][5]  # taskDelay
                # print('time_task is :', time_task)
                test_capacity_task = test_list_now[test_a][6]
                test_c = v

            else:
                if test_task_remain < 0:
                    test_task_success, test_ratio_0 = test_env.judgement(test_list_now, test_process_number, test_package_task)
                    test_a = agent.act(test_state)
                    list_action.append([test_a])
                    if test_a >= len(test_list_now):  ### 123456 are all ok
                        test_state_old = test_state
                        test_test_flag = True
                        continue
                    test_state_old = test_state
                    test_process_number = test_list_now[test_a][0]
                    test_package_task = test_list_now[test_a][4]
                    test_task_remain = test_package_task
                    test_time_task = test_list_now[test_a][5]  # new
                    test_capacity_task = test_list_now[test_a][6]
                    test_c = v

                else:
                    test_circle_out, test_aa = test_env.judge(test_list_now, test_process_number, test_package_task)
                    if test_circle_out is True:
                        test_a = agent.act(test_state)
                        list_action.append([test_a])
                        if test_a >= len(test_list_now):  ### 123456 are all ok
                            test_state_old = test_state
                            test_test_flag = True
                            continue
                        test_state_old = test_state
                        test_process_number = test_list_now[test_a][0]
                        test_package_task = test_list_now[test_a][4]
                        test_task_remain = test_package_task  # package_task will be unchanged even though task_remain cha
                        test_time_task = test_list_now[test_a][5]  # new
                        test_capacity_task = test_list_now[test_a][6]
                        test_c = v

                    else:
                        test_task_remain = (test_task_remain - test_capacity_task * test_tc)
                        test_capacity_task = test_list_now[test_aa][6]

        test_r_success_selected, test_r_total_size, test_r_success_select_bit, test_r_success_total_bit \
            = test_env.ratio_result(test_sum_package1)
    return test_r_success_total_bit


if __name__== "__main__":
    r_c = 1000
    r_t = 500
    r_u = 740  # radius of UAV flying
    V = 90  # m/s
    B = 1   #M
    P = 10**2   # mW
    A = 10**(-7)  # W
    N = 10**(-174/10)*B*(10**6)  # mW  #######################
    p_min = 7  # min package size
    p_max = 15  # max package size
    batch_size = 32
    test_flag = False     ### if chose(predict) padding 0, it is True
    reward_memory = deque(maxlen=10)
    list_action = []
    list_random = []
    list_nearest = []
    list_smallest = []
    list_furthest = []
    list_furthest_p = []

    list_random_success_select_bit = []
    list_nearest_success_select_bit = []
    list_smallest_success_select_bit = []
    list_furthest_success_select_bit = []
    list_furthest_p_success_select_bit = []
    list_NN_success_select_bit = []

    list_random_success_total_bit = []
    list_nearest_success_total_bit = []
    list_smallest_success_total_bit = []
    list_furthest_success_total_bit = []
    list_furthest_p_success_total_bit = []
    list_NN_success_total_bit = []
    test_list_NN_success_total_bit = []

    list_random_total_size = []
    list_nearest_total_size = []
    list_smallest_total_size = []
    list_furthest_total_size = []
    list_furthest_p_total_size = []

    list_MTs_density = []
    list_episodes = []
    list_epsilon = []
    list_Q_value = []
    number_test_record = []
    number_test = 0
    list_number_chose0 = []
    list_all_q = []

    #print('1')
    method = 10   ####### 10 for NN,   or 0 for other methods
    network_Model = 0  # 0: NN, 1: load   2: CNN
    episodes = 300  #### for method 10:  episodes=10:memory=292(around)
    EPISODES = 50 ## for test
    circles = 5     #### for other methods
    len_memory = 0

    if method == 10:
        MTs_density = 30
        density_value = 10
        reward = 0

        #random.seed(11)
        env = Environment()
        user = Users()
        agent = Agent(density_value, network_Model)

        for l in range(episodes):
            random.seed(random_Seed) #random the same data every episodes
            env.reset()
            agent.reset()
            number_chose0 = 0
            Ts = (2 * math.pi * r_u) / V  # time of flying a circle by UAV
            tc = 0.5  # decision time
            counter = math.ceil(Ts / tc)  # one circle has number of counter operations ### 104
            print('counter is:', counter)

            theta_UAV = 0
            theta_user = 0
            list_in = []
            list_now = []

            package_task = 0
            time_task = 0
            capacity_task = 0
            task_remain = 0
            c = 0

            success_bit = 0
            total_bit = 0

            list_in1, sum_package1 = user.initial_coordinate()
            total_size = len(list_in1)
            print('all users in the ring:', len(list_in1))  # all of the users in the ring 500~1000

            t0 = 0
            print('t0 is :', t0)

            for m in range(counter):
                list_now0 = env.list_update(m, total_size,
                                           list_in1)  # every tc=0.5, update the UAV's location and list_now

                action_size0 = len(list_now0)
                if action_size0 == 0:
                    continue
                print('len of list_now is:', len(list_now0))
                state, list_now, action_size = env.getState(list_now0, action_size0)
                #print('state is:', state)

                if m == 0 or test_flag is True:
                    if test_flag is True:
                        agent.store_transition(state_old, a, reward, state)
                        len_memory += 1

                    a = agent.act(state)

                    if a >= len(list_now):  ### chose padding 0
                        reward = -1
                        reward = agent.longterm_reward_compute(reward)
                        number_chose0 += 1
                        state_old = state
                        test_flag = True
                        continue

                    test_flag = False
                    state_old = state
                    process_number = list_now[a][0]  # serial number
                    #print('chosen task is:', process_number)
                    package_task = list_now[a][4]
                    task_remain = package_task  # after act, task_remain equals package_task
                    time_task = list_now[a][5]  # taskDelay
                    #print('time_task is :', time_task)
                    capacity_task = list_now[a][6]
                    c = m

                else:
                    if task_remain < 0:
                        #print('len of list_now before judgement is:', len(list_now))
                        task_success, ratio_0 = env.judgement(list_now, process_number, package_task)
                        print('ratio after judgement is:', ratio_0)

                        if task_success is True:
                            reward = (package_task - p_min) / (p_max - p_min)
                            reward = agent.longterm_reward_compute(reward)
                        else:
                            task_remain = (task_remain + capacity_task * tc)  # compute last task_remain
                            reward = -(task_remain - p_min) / (p_max - p_min)
                            reward = agent.longterm_reward_compute(reward)
                        agent.store_transition( state_old, a, reward, state)
                        len_memory += 1
                        a = agent.act(state)
                        print('!!!!!!!', a)
                        #if list_now[a][3] == 0:
                        if a >= len(list_now):   ### chose padding 0
                            reward = -1
                            reward = agent.longterm_reward_compute(reward)
                            number_chose0 += 1
                            state_old = state
                            test_flag = True
                            continue
                        state_old = state
                        process_number = list_now[a][0]
                        #print('chosen task is:', process_number)
                        package_task = list_now[a][4]
                        task_remain = package_task
                        time_task = list_now[a][5]  # new
                        #print('time_task is :', time_task)
                        capacity_task = list_now[a][6]
                        c = m

                    else:
                        #print('len of list_now before judge is:', len(list_now))
                        circle_out, aa = env.judge(list_now, process_number, package_task)
                        print('circle_out is:', circle_out)
                        if circle_out is True:
                            reward = -(task_remain - p_min) / (p_max - p_min)
                            reward = agent.longterm_reward_compute(reward)
                            agent.store_transition( state_old, a, reward, state)
                            len_memory += 1
                            a = agent.act(state)
                            print('!!!!!!!', a)
                            if a >= len(list_now):  ### 123456 are all ok
                                reward = -1
                                reward = agent.longterm_reward_compute(reward)
                                number_chose0 += 1
                                state_old = state
                                test_flag = True
                                continue
                            state_old = state
                            process_number = list_now[a][0]
                            #print('chosen task is:', process_number)
                            package_task = list_now[a][4]
                            task_remain = package_task   #package_task will be unchanged even though task_remain cha
                            time_task = list_now[a][5]  # new
                            #print('time_task is :', time_task)
                            capacity_task = list_now[a][6]
                            c = m

                        else:
                            task_remain = (task_remain - capacity_task * tc)
                            capacity_task = list_now[aa][6]

                print('len of  memory is:', len_memory)

                if len_memory  > batch_size:
                    agent.replay(batch_size)

                if ((m % EPISODES) == 0) and (m != 0):   # for test
                    print('%%%%%%%%%%%*************')
                    agent.exploration = False
                    number_test += 0.5
                    number_test_record.append(number_test)    # figure 10
                    test_r_success_total_bit1 = test_for_real_success_total_bit(agent)
                    test_list_NN_success_total_bit.append([test_r_success_total_bit1])   # figure 10
                    print('************************')
                    agent.exploration = True

            r_success_selected, r_total_size, r_success_select_bit, r_success_total_bit \
                = env.ratio_result(sum_package1)
            list_NN_success_select_bit.append([r_success_select_bit])

            list_NN_success_total_bit.append([r_success_total_bit])   # figure 10
            list_number_chose0.append([number_chose0])   # figure 13

            list_epsilon.append([agent.epsilon])
            list_episodes.append([l])

        Q_NN, number_Q_NN = agent.Q_value_compute()    # figure 11
        print(Q_NN, number_Q_NN)
        test_Q_NN, test_number_Q_NN = agent.test_Q_value_compute()   # figure 11
        print(test_Q_NN, test_number_Q_NN)
        print(list_episodes, list_epsilon)

        loss_NN, number_reply_NN = agent.loss_compute()   # figure 12

        agent.save()

        fig10 = plt.figure(10)
        plt.plot(list_episodes, list_NN_success_total_bit, color='red', label='NN_train')
        plt.plot(number_test_record, test_list_NN_success_total_bit, color='yellow', label='NN_test')
        plt.legend()
        plt.xlabel('episodes')
        plt.ylabel('bit: success/total')

        fig11 = plt.figure(11)
        plt.plot(number_Q_NN, Q_NN, color='red', label='NN_train')
        plt.plot(test_number_Q_NN, test_Q_NN, color='yellow', label='NN_test')
        plt.legend()
        plt.xlabel('times of act by NN')
        plt.ylabel('Q_value')

        fig12 = plt.figure(12)
        plt.plot(number_reply_NN, loss_NN, color='red', label='NN')
        plt.legend()
        plt.xlabel('times of reply')
        plt.ylabel('loss')

        fig121 = plt.figure(121)
        plt.plot(list_all_q, color='green', label='NN')
        plt.legend()
        plt.xlabel('10 times of reply')
        plt.ylabel('act_values')

        fig13 = plt.figure(13)
        plt.plot(list_episodes, list_number_chose0, color='black', label='times of choosing 0')
        plt.plot(list_episodes, list_epsilon, color='red', label='epsilon', marker = '*')
        plt.legend()
        plt.xlabel('episodes')
        plt.ylabel('times of choosing 0')

        fig14 = plt.figure(14)
        plt.plot(list_action, color='yellow', label='action index')
        plt.legend()
        plt.xlabel('episodes')
        plt.ylabel('action index')
        plt.show()

    else:
        for MTs_density in range(25, 250, 50):  # ##################100:400users
            for method in range(5):
                ratio_success_selected = 0
                ratio_total_size = 0
                ratio_success_select_bit = 0
                ratio_success_total_bit = 0
                random.seed(11)

                for i in range(circles):
                    Ts = (2 * math.pi * r_u) / V  # time of flying a circle by UAV
                    tc = 0.5  # decision time
                    counter = math.ceil(Ts / tc)  # one circle has number of counter operations

                    sum_package = 0

                    env = Environment()
                    user = Users()
                    ratio = 0

                    theta_UAV = 0
                    theta_user = 0
                    list_in = []
                    list_now = []

                    package_task = 0
                    time_task = 0
                    capacity_task = 0
                    task_remain = 0
                    c = 0

                    success_bit = 0
                    total_bit = 0

                    list_in1, sum_package1 = user.initial_coordinate()
                    total_size = len(list_in1)
                    print('list_in1 is:', len(list_in1))  # all of the users in the ring 500~1000

                    t0 = 0
                    print('t0 is :', t0)

                    for j in range(counter):
                        list_now = env.list_update(j, total_size,
                                                   list_in1)  # every tc=0.5, update the UAV's location and list_now

                        action_size = len(list_now)
                        if action_size == 0:
                            continue
                        print('len of list_now is:', len(list_now))
                        print(list_now)  # ## pointer list_now , list_now1, list_now2 are all the same

                        if j == 0:
                            agent = Agent(action_size, network_Model)
                            a = agent.act(list_now)
                            process_number = list_now[a][0]  # serial number
                            print('chosen task is:', process_number)
                            package_task = list_now[a][4]
                            task_remain = package_task  # after act, task_remain equals package_task
                            time_task = list_now[a][5]  # taskDelay
                            print('time_task is :', time_task)
                            capacity_task = list_now[a][6]
                            c = j

                        else:
                            if task_remain < 0:
                                print('len of list_now before judgement is:', len(list_now))
                                task_success, ratio_0 = env.judgement(list_now, process_number, package_task)
                                print('ratio after judgement is:', ratio_0)

                                agent = Agent(action_size, network_Model)
                                a = agent.act(list_now)
                                process_number = list_now[a][0]
                                print('chosen task is:', process_number)
                                package_task = list_now[a][4]
                                task_remain = package_task
                                time_task = list_now[a][5]  # new
                                print('time_task is :', time_task)
                                capacity_task = list_now[a][6]
                                c = j

                            else:
                                print('len of list_now before judge is:', len(list_now))
                                circle_out, aa = env.judge(list_now, process_number, package_task)
                                print('circle_out is:', circle_out)
                                if circle_out is True:
                                    agent = Agent(action_size, network_Model)
                                    a = agent.act(list_now)
                                    process_number = list_now[a][0]
                                    print('chosen task is:', process_number)
                                    package_task = list_now[a][4]
                                    task_remain = package_task
                                    time_task = list_now[a][5]  # new
                                    print('time_task is :', time_task)
                                    capacity_task = list_now[a][6]
                                    c = j

                                else:
                                    task_remain = (task_remain - capacity_task * tc)
                                    capacity_task = list_now[aa][6]

                    r_success_selected, r_total_size, r_success_select_bit, r_success_total_bit \
                        = env.ratio_result(sum_package1)

                    ratio_success_selected += r_success_selected
                    ratio_total_size += r_total_size
                    ratio_success_select_bit += r_success_select_bit
                    ratio_success_total_bit += r_success_total_bit

                ratio_success_selected = ratio_success_selected / circles
                ratio_total_size = ratio_total_size / circles
                ratio_success_select_bit = ratio_success_select_bit / circles
                ratio_success_total_bit = ratio_success_total_bit / circles

                print('final ratio is:', ratio_success_selected, 'ratio_total_size is:', ratio_total_size,
                      'ratio_success_select_bit is:', ratio_success_select_bit, 'ratio_success_total_bit is: ',
                      ratio_success_total_bit)

                if method == 0:
                    list_random.append([ratio_success_selected])
                    list_random_total_size.append([ratio_total_size])
                    list_random_success_select_bit.append([ratio_success_select_bit])
                    list_random_success_total_bit.append([ratio_success_total_bit])

                elif method == 1:
                    list_nearest.append([ratio_success_selected])
                    list_nearest_total_size.append([ratio_total_size])
                    list_nearest_success_select_bit.append([ratio_success_select_bit])
                    list_nearest_success_total_bit.append([ratio_success_total_bit])

                elif method == 2:
                    list_smallest.append([ratio_success_selected])
                    list_smallest_total_size.append([ratio_total_size])
                    list_smallest_success_select_bit.append([ratio_success_select_bit])
                    list_smallest_success_total_bit.append([ratio_success_total_bit])

                elif method == 3:
                    list_furthest.append([ratio_success_selected])
                    list_furthest_total_size.append([ratio_total_size])
                    list_furthest_success_select_bit.append([ratio_success_select_bit])
                    list_furthest_success_total_bit.append([ratio_success_total_bit])

                elif method == 4:
                    list_furthest_p.append([ratio_success_selected])
                    list_furthest_p_total_size.append([ratio_total_size])
                    list_furthest_p_success_select_bit.append([ratio_success_select_bit])
                    list_furthest_p_success_total_bit.append([ratio_success_total_bit])

                else:
                    print('method cannot be found')

            list_MTs_density.append([MTs_density])  ################################################

        fig1 = plt.figure(1)
        plt.plot(list_MTs_density, list_random, color='green', label='random')
        plt.plot(list_MTs_density, list_nearest, color='red', label='nearest')
        plt.plot(list_MTs_density, list_smallest, color='blue', label='smallest')
        plt.plot(list_MTs_density, list_furthest, color='black', label='furthest')
        plt.plot(list_MTs_density, list_furthest_p, color='yellow', label='furthest_p')
        plt.legend()

        plt.xlabel('MTs_density')
        plt.ylabel('ratio: success/selected')

        fig2 = plt.figure(2)
        plt.plot(list_MTs_density, list_random_total_size, color='green', label='random')
        plt.plot(list_MTs_density, list_nearest_total_size, color='red', label='nearest')
        plt.plot(list_MTs_density, list_smallest_total_size, color='blue', label='smallest')
        plt.plot(list_MTs_density, list_furthest_total_size, color='black', label='furthest')
        plt.plot(list_MTs_density, list_furthest_p_total_size, color='yellow', label='furthest_p')
        plt.legend()

        plt.xlabel('MTs_density')
        plt.ylabel('ratio: success/total')

        fig3 = plt.figure(3)
        plt.plot(list_MTs_density, list_random_success_select_bit, color='green', label='random')
        plt.plot(list_MTs_density, list_nearest_success_select_bit, color='red', label='nearest')
        plt.plot(list_MTs_density, list_smallest_success_select_bit, color='blue', label='smallest')
        plt.plot(list_MTs_density, list_furthest_success_select_bit, color='black', label='furthest')
        plt.plot(list_MTs_density, list_furthest_p_success_select_bit, color='yellow', label='furthest_p')
        plt.legend()

        plt.xlabel('MTs_density')
        plt.ylabel('bit: success/selected')

        fig4 = plt.figure(4)
        plt.plot(list_MTs_density, list_random_success_total_bit, color='green', label='random')
        plt.plot(list_MTs_density, list_nearest_success_total_bit, color='red', label='nearest')
        plt.plot(list_MTs_density, list_smallest_success_total_bit, color='blue', label='smallest')
        plt.plot(list_MTs_density, list_furthest_success_total_bit, color='black', label='furthest')
        plt.plot(list_MTs_density, list_furthest_p_success_total_bit, color='yellow', label='furthest_p')
        plt.legend()

        plt.xlabel('MTs_density')
        plt.ylabel('bit: success/total')
        plt.show()








