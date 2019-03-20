format: python_code_name(result_profile_name), comment_code in the last code: main


DQN_1(main1_5):
 1:epsilon = 0.9996   exploration condition!
 2:act while
 3:Q value plotted as action-number
 4:plot select 0
 5:Q value of test 

  modification: atan2!



DoubleDQN_2(main1_6):
  6.DoubleDQN                             



DoubleDQN_CNN_3(main1_7):
   7. CNN



DoubleDQN_LTreward_4(main1_8): NN double
   8: long term reward£ºmean of last ten times reward       loss decrease...
      modification:act, random.seed(random_Seed), plot action-number of test

    modication: reply index£¡

  Note:
      proper:  number of circles:300  deque:5000 
     300NNatan2£º2000£¬32
     5000£º      5000£¬ 32
     64£º        5000£¬ 64
     r0:         5000, 64,  random can be 0!
     r032:       5000,  32
     0.9995:     5000, 32, epsilon_decay = 0.9995
     0.9993:     5000, 32, epsilon_decay = 0.9993

     actreal:    6000, 32   random cannot select 0!
     0.0005:     6000, 32, epsilon_decay = 0.9993, learning rate:0.0005, 400 circles
     0.0008£º                                                    0.0008
     16£º        6000£¬ 16£¬       0.9993£¬                      0.0008£¬ 300
     16compare   5000£¬ 16£¬       0.9993£¬                      0.001£¬  300
     32compare          32
     allq:       plot all Q value
     fail£º-1£¬ -2




DoubleDQN_LTreward_test_5(main1_10) from main1_8:
    test when target network is updated
    fail: reward=-1 no zero initialization!



Dueling_DDQN_6 (main1_11) from main1_8 :
     
   1:NN£¬ not double DQN£¬ long term reward
   2:NN,  double,  long term reward
    fail: reward = -1
    01£ºreward=-1 when failed£¬reward=-2 when chose 0


PER_DQN_7(main1_12) from main1_8:
    P: Prioritized DQN
    ini: zero initialization
    fail: reward=-1 when failed£¬reward=-2 when chose 0


(main1_14) from main1_8:
   Measurement: success bits/total bits
   fail:reward=-1 when failed
   ini:  zero initialization
   fail500
   01£ºreward=-2 when chose 0
   02: random put outside £¬different data every circle



DoubleDQN_LTreward_measure_8(main1_15) from main1_8:
   Measurement: success number/total number
   fail: reward=-1 when failed,   reward=1 when successed
   01:reward=-1 when failed
   02:test when target network is updated
   03£ºreward=-1 when failed£¬reward=-2 when chose 0



main:  Comment code!!!!!
  all the methods used in: Double, Dueling, PER
  ÎÞ£º
  200£ºcircle_number is 200
  ir:no random