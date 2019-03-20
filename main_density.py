import numpy as np
import matplotlib.pylab as plt

import random
from collections import deque
import scipy

import threading
import time

from apscheduler.schedulers.blocking import BlockingScheduler

import math

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
random_Seed = 100

class Users:
    def __init__(self) :
        usernum = 100

    def initial_coordinate(self):
        ran = ((2*r_c)**2)*MTs_density/(10**6)
        print(ran)
        sum_package = 0
        for i in range(0,int(ran)):
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
        theta_UAV = (time_list-t0)*V/self.r_u    # unit is rad
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

class Agent:
    def __init__(self, action_size1, network_model):
        self.state_size = 10*3      #############
        self.action_size = action_size1
        #print('action size is:', action_size1)
        self.memory = deque(maxlen=5000)
        self.gamma = 0.9      # decay rate
        self.epsilon = 1.0     # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.loss = 0
        self.loss_record = []
        self.number_reply_record = []
        self.exploration = True
        # if network_model == 0:
        #     self.model = self._build_NN_model()
        # if network_model == 1:
        #     self.model = load_model(filepath="model.h5")
        #     self.exploration = False
        self.Q_value = 0
        self.number_reply = 0

    def reset(self):
        print('agent reset')
        self.Q_value = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, list_now4):
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

        else:
            print('method cannot find')


if __name__== "__main__":
    r_c = 1000
    r_t = 500
    r_u = 740
    V = 90
    B = 1   #M
    P = 10**2   # mW
    A = 10 ** (-7)  # W
    N = 10 ** (-174 / 10) * B * (10 ** 6)  # mW  #######################
    p_min = 7  # min package size
    p_max = 15  # max package size
    batch_size = 32
    test_flag = False  ### if chose(predict) padding 0, it is True
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

    list_random_success_total_bit = []
    list_nearest_success_total_bit = []
    list_smallest_success_total_bit = []
    list_furthest_success_total_bit = []
    list_furthest_p_success_total_bit = []

    list_random_total_size = []
    list_nearest_total_size = []
    list_smallest_total_size = []
    list_furthest_total_size = []
    list_furthest_p_total_size = []

    list_MTs_density = []
    network_Model = 0  # 0: NN, 1: load
    circles = 5

    print('1')
    for MTs_density in range(30, 110, 10):  # ##################100:400users
        for method in range(5):
            ratio_success_selected = 0
            ratio_total_size = 0
            ratio_success_select_bit = 0
            ratio_success_total_bit = 0
            random.seed(random_Seed)

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

            ratio_success_selected = ratio_success_selected/circles
            ratio_total_size = ratio_total_size/circles
            ratio_success_select_bit = ratio_success_select_bit/circles
            ratio_success_total_bit = ratio_success_total_bit/circles

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

        list_MTs_density.append([MTs_density])    ################################################

    print(list_random)
    print(list_random_total_size)
    print(list_random_success_select_bit)
    print(list_random_success_total_bit)

    print(list_nearest)
    print(list_nearest_total_size)
    print(list_nearest_success_select_bit)
    print(list_nearest_success_total_bit)

    print(list_smallest)
    print(list_smallest_total_size)
    print(list_smallest_success_select_bit)
    print(list_smallest_success_total_bit)

    print(list_furthest)
    print(list_furthest_total_size)
    print(list_furthest_success_select_bit)
    print(list_furthest_success_total_bit)

    print(list_furthest_p)
    print(list_furthest_p_total_size)
    print(list_furthest_p_success_select_bit)
    print(list_furthest_p_success_total_bit)

    fig1=plt.figure(1)
    plt.plot(list_MTs_density, list_random, color='green', label='random')
    plt.plot(list_MTs_density, list_nearest, color='red', label='nearest')
    plt.plot(list_MTs_density, list_smallest, color='blue', label='smallest')
    plt.plot(list_MTs_density, list_furthest, color='black', label='furthest')
    plt.plot(list_MTs_density, list_furthest_p, color='yellow', label='furthest_p')
    plt.legend()

    plt.xlabel('MTs_density')
    plt.ylabel('ratio: success/selected')

    fig2=plt.figure(2)
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







