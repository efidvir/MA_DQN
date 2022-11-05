# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import gym
from gym import spaces
import cv2
#from google.colab.patches import cv2_imshow
#from google.colab import output
#import time
import os, sys
#os.environ["SDL_VIDEODRIVER"] = "dummy"
import matplotlib.pyplot as plt
#plt.rcParams["figure.dpi"] = 300
from matplotlib import colors
#import networkx as nx
#from networkx.drawing.nx_agraph import write_dot
#from networkx.drawing.nx_pydot import write_dot

#import pygame
#from sklearn.preprocessing import normalize
#import graphviz
#from graphviz import Source
#np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))
#import ffmpeg
#import moviepy.video.io.ImageSequenceClip
np.set_printoptions(precision=4)
from agents import Q_transmit_agent
from agents import AC_Agent
from agents import DQN_transmit_agent
from env import transmit_env
from visualize import render
draw = render()

#agent type
#agent_type = 'Actor-Critic'
#agent_type = 'Q_Learning'
agent_type = 'DQN'

#Global parameters
number_of_iterations = 5000000
force_policy_flag = True
number_of_agents = 100
np.random.seed(0)

#model
MAX_SILENT_TIME = 200
SILENT_THRESHOLD = 1
BATTERY_SIZE = 200
DISCHARGE = 99
MINIMAL_CHARGE = 0
CHARGE = 1
number_of_actions = 2

#learning params
GAMMA = 0.9
ALPHA = 0.01
#P_LOSS = 0
decay_rate = 0.999999

#for rendering
DATA_SIZE = 10

'''run realtime experiences'''
#T = [[] for i in range(number_of_agents)]
#for i in range(number_of_agents):
#    T[i] = np.zeros(shape=(BATTERY_SIZE * MAX_SILENT_TIME, MAX_SILENT_TIME * BATTERY_SIZE))  # transition matrix
policies = [[] for i in range(number_of_agents)]
values = [[] for i in range(number_of_agents)]
#pol_t = np.ndarray(shape=(number_of_iterations, number_of_agents, BATTERY_SIZE, MAX_SILENT_TIME))
#val_t = np.ndarray(shape=(number_of_iterations, number_of_agents, BATTERY_SIZE, MAX_SILENT_TIME))

occupied = 0
epsilon = np.ones(number_of_agents)
print(epsilon)
# initialize environment
env = [[] for i in range(number_of_agents)]
agent = [[] for i in range(number_of_agents)]
state = [[] for i in range(number_of_agents)]
actions = [[] for i in range(number_of_agents)]
transmit_or_wait_s = [[] for i in range(number_of_agents)]
score = [[] for i in range(number_of_agents)]
RAND = [[np.random.randint(10000)] for i in range(number_of_agents)]

for i in range(number_of_agents):
    #epsilon[i] = epsilon[i] -1/(number_of_agents+i)
    env[i] = transmit_env(BATTERY_SIZE, MAX_SILENT_TIME, (i+3), MINIMAL_CHARGE, DISCHARGE, CHARGE, DATA_SIZE, number_of_actions)
    if agent_type == 'Q_Learning':
        agent[i] = Q_transmit_agent(ALPHA, GAMMA, BATTERY_SIZE, MAX_SILENT_TIME, DATA_SIZE, number_of_actions, MINIMAL_CHARGE,RAND[i])
        Q_tables = [[] for i in range(number_of_iterations)]
    elif agent_type == 'Actor-Critic':
        agent[i] = AC_Agent(5*i*0.0000008, GAMMA, BATTERY_SIZE, MAX_SILENT_TIME, DATA_SIZE, number_of_actions,MINIMAL_CHARGE)
        print('Make sure to adjust the learning rate')
    elif agent_type == 'DQN':
        agent[i] = DQN_transmit_agent(ALPHA, GAMMA, BATTERY_SIZE, MAX_SILENT_TIME, DATA_SIZE, number_of_actions, MINIMAL_CHARGE,RAND[i])
    state[i] = env[i].initial_state
    actions[i] , transmit_or_wait_s[i] = agent[i].choose_action(state[i], epsilon[i])
    #policies[i] = agent[i].get_policy()
    #values[i] = agent[i].get_state_value(policies[i])
# pol_t[0] = policies
# val_t[0] = values

# plot reward function in use
#plt.plot(range(len(env[0].r_1)), env[0].r_1, 'o--', color='blue')
#plt.xticks(range(env[0].max_silence_time))
#plt.title('Reward function $r_1$')
#plt.show(block=False)
print(epsilon)
print('r_1 array: ', env[0].r_1)

# data = np.zeros(env.data_size)
errors = [[] for i in range(number_of_agents)]

for i in range(number_of_iterations):

    # all agents move a step and take a new action
    for j in range(number_of_agents):
        env[j].state = env[j].new_state
    # Gateway decision
    if sum(transmit_or_wait_s) > 1 or sum(transmit_or_wait_s) == 0:
        ack = 0
    elif sum(transmit_or_wait_s) == 1:
        ack = 1

    for j in range(number_of_agents):
        new_state, reward, occupied = env[j].time_step(actions[j], transmit_or_wait_s[j], sum(transmit_or_wait_s), ack)  # CHANNEL
        # errors.append(np.mean(agent.error))
        env[j].new_state = new_state
        score[j].append(reward)
        np.random.seed(j)
        #print('Agent ', j)
        actions[j], transmit_or_wait_s[j] = agent[j].step(env[j].state, reward, actions[j], transmit_or_wait_s[j], env[j].new_state, epsilon[j])

    if i % 100 == 0:
        print('step: ', i, '100 steps AVG mean score: ',np.mean(score[0][-100:-1]))

    # collect state transitions in T
    for j in range(number_of_agents):
        #print(env[j].state, env[j].new_state)
        # decompose state
        #current_energy, slient_time = env[j].state
        # decompose new state
        #next_energy, next_silence = env[j].new_state
        # print(current_energy, slient_time,'->',next_energy, next_silence , '~~~', current_energy*(BATTERY_SIZE-1)+slient_time, next_energy*(BATTERY_SIZE-1)+next_silence)
        #T[j][current_energy * (BATTERY_SIZE) + slient_time, next_energy * (BATTERY_SIZE) + next_silence] += 1
        epsilon[j] = epsilon[j] * decay_rate
        #draw.render_Q_diffs(agent[j].Q[:, :, 0], agent[j].Q[:, :, 1], j,i,env[j].state)
    #Q_tables[i] = np.array(agent[0].Q[:][:][0])
    #render_policy_visits_table(get_policy(0), agent[0].state_visits)

#for j in range(number_of_agents):
#    print('video done')
#    draw.render_Q_diffs_video(agent[j].Q[:, :, 0], agent[j].Q[:, :, 1], j,number_of_iterations)


# env.render(data)
print(epsilon)
# plt.plot(errors)
#video.release()

#print('policy: ',policies)

#Agent evaluation

# No exploration
epsilon = np.zeros(number_of_agents)

data = []
collisions = 0
agent_clean = [np.zeros(1) for i in range(number_of_agents)]
wasted = 0

num_of_eval_iner = 1000
#Teval = [[] for i in range(number_of_agents)]
#for a in range(number_of_agents):
#    Teval[a] = np.zeros(shape=(BATTERY_SIZE * MAX_SILENT_TIME, MAX_SILENT_TIME * BATTERY_SIZE))  # transition matrix

for i in range(num_of_eval_iner):
    for a in range(number_of_agents):
        env[a].state = env[a].new_state

    # Gateway decision
    if sum(transmit_or_wait_s) > 1 or sum(transmit_or_wait_s) == 0:
        ack = 0
    elif sum(transmit_or_wait_s) == 1:
        ack = 1

    if sum(transmit_or_wait_s) > 1:
        collisions += 1
        data.append(1)
    if sum(transmit_or_wait_s) == 1:
        for a in range(number_of_agents):
            if transmit_or_wait_s[a] == 1:
                agent_clean[a] += 1
                data.append(a+2)
    if sum(transmit_or_wait_s) == 0:
        wasted += 1
        data.append(0)
    for a in range(number_of_agents):
        new_state, reward, occupied = env[a].time_step(actions[a],transmit_or_wait_s[a], sum(transmit_or_wait_s), ack)  # CHANNEL
        env[a].new_state = new_state

    for a in range(number_of_agents):
        # print(env.state, env,new_state)
        actions[a] ,transmit_or_wait_s[a] = agent[a].step(env[a].state, reward, actions[a],transmit_or_wait_s[a], env[a].new_state, epsilon[a])

    for a in range(number_of_agents):
        # decompose state
        current_energy, slient_time = env[a].state
        # decompose new state
        next_energy, next_silence = env[a].new_state
        # print(current_energy, slient_time,'->',next_energy, next_silence , '~~~', current_energy*(BATTERY_SIZE-1)+slient_time, next_energy*(BATTERY_SIZE-1)+next_silence)
        #Teval[a][current_energy * (BATTERY_SIZE) + slient_time, next_energy * (BATTERY_SIZE) + next_silence] += 1
print('collisions', collisions)
for a in range(number_of_agents):
    print('agent{d}'.format(d=a), agent_clean[a])
    plt.plot(range(len(score[a])), score[a])
    plt.show(block=False)
    #print(agent[a].state_visits)
print('wasted', wasted)
print(data)
draw.last_1k_slots(data, number_of_agents)
'''
for i in range(number_of_agents):
    print('Agent ', i)
    print('\n')

    #draw.plot_Q_values(Q_tables,number_of_iterations)

for i in range(number_of_agents):
    print('Agent ',i,' Q table:', Q_tables[i])
    draw.render_Q(agent[j].Q[:, :, 0], agent[j].Q[:, :, 1], j, i, env[j].state)
    cv2.waitKey(0)
'''