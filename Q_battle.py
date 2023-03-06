#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:53:53 2023

@author: sergej
"""
# =============================================================================
# Q-learning playing comp 2x2 against it self
# =============================================================================
import numpy as np
from random import choice
from itertools import combinations_with_replacement

# Set game parameters
g_small = 1
g_big = 2
c_small = -1
c_big = -2
payoffs = list(combinations_with_replacement([g_small, c_small, c_big], 2))[:-1]
pSuccess_range = np.arange(0.1, 1, 0.1)

# Game rules of stochastic prisoner's dilemma
def filterPD():
    weather_types = []
    for i in range(0, len(pSuccess_range)):
        for j in range(i+1, len(pSuccess_range)):
            pS = pSuccess_range[i]
            pT = pSuccess_range[j]
            pP = (pT*(1-pT))+(pT*pT)/2
            pR = 1-(1-pS)**2
            R = pR*g_small+(1-pR)*c_small
            S = pS*g_small+(1-pS)*c_big
            T = pT*g_small+(1-pT)*c_small
            P = pP*g_small+(1-pP)*c_small
            if T > R > P > S and 2*R > T+S and 0.4 <= pT < 0.8 and pS < 0.4:
                weather_types.append({"pR":pR, "pS":pS, "pT":pT, "pP":pP,
                                      "R":R, "S":S, "T":T, "P":P}) 
    return weather_types
weather_types = [filterPD()[0], filterPD()[2]]

# Define state matrix size and action matrix size
blck_n = 25 # Nr. of repetitions before exploration decay
days_n = 2 # Nr. of timpe-points
states_n = 4  # 0 life points, 1 life point, 2 life points, 3 life points
actions_n = 2  # 0: near forest, 1: far forest
weather_n = 2  # 0: weather1, 1: weather2

# Set hyper parameters
episodes = 10000
learning_rate = 0.8
discount_factor = 0.95
exploration_rate = 1
exploration_decay = 0.99

# Initialize Q-tables for 2 players
q_table1 = np.zeros((weather_n, actions_n, days_n+1, states_n, states_n))
q_table2 = np.zeros((weather_n, actions_n, days_n+1, states_n, states_n))

# Define function to get probability of payoff
def prob(action1, action2, weather_i, state1, state2):
    # Self not dead
    if state1 != 0:
        # Opponent not dead
        if state2 != 0:
            if action1 == "far" and action2 == "far":
                return weather_types[weather_i]["pR"]
            elif action1 == "far" and action2 == "near":
                return weather_types[weather_i]["pS"]
            elif action1 == "near" and action2 == "far":
                return weather_types[weather_i]["pT"]
            else:
                return weather_types[weather_i]["pP"]
        # Opponent dead
        else:
            if action1 == "near":
                return weather_types[weather_i]["pT"]
            else:
                return weather_types[weather_i]["pS"]
    # Self dead
    else:
        return 0

# Define function to calculate payoffs
def get_payoffs(state1, state2, action1, action2, p_success1, p_success2):
    global transition_mags
    succ1 = np.random.uniform(0, 1) <= p_success1
    succ2 = np.random.uniform(0, 1) <= p_success2
    # Both players alive
    if state1 != 0 and state2 != 0:
        # If both players choose "far"
        if action1 == action2 == "far":
            if succ1 == True:
                transition_mags = list(payoffs[0])
            else:
                transition_mags = list(payoffs[-2])
        # If player1 chooses "far" and player2 chooses "near"
        elif action1 == "far" and action2 == "near":
            if succ1 == True and succ2 == True:
                transition_mags = list(payoffs[0])
            elif succ1 == True and succ2 == False:
                transition_mags = list(payoffs[1])
            elif succ1 == False and succ2 == True:
                transition_mags = list(payoffs[2][::-1])
            elif succ1 == False and succ2 == False:
                transition_mags = list(payoffs[-1][::-1])
        # If player1 chooses "near" and player2 chooses "far"
        elif action1 == "near" and action2 == "far":
            if succ1 == True and succ2 == True:
                transition_mags = list(payoffs[0])
            elif succ1 == True and succ2 == False:
                transition_mags = list(payoffs[2])
            elif succ1 == False and succ2 == True:
                transition_mags = list(payoffs[1][::-1])
            elif succ1 == False and succ2 == False:
                transition_mags = list(payoffs[-1])
        # If pboth players choose "near"
        else:
            if succ1 == True and succ2 == True:
                transition_mags = list(payoffs[0])
            elif succ1 == True and succ2 == False:
                transition_mags = list(payoffs[1])
            elif succ1 == False and succ2 == True:
                transition_mags = list(payoffs[1][::-1])
            elif succ1 == False and succ2 == False:
                transition_mags = list(payoffs[-2])     
    # Player1 alive, player2 dead
    elif state1 != 0 and state2 == 0:
        # Player1 chooses "far"
        if action1 == "far":
            if succ1 == True:
                transition_mags = [g_small, 0]
            else:
                transition_mags = [c_big, 0]
        # Player1 chooses "near"
        if action1 == "near":
            if succ1 == True:
                transition_mags = [g_small, 0]
            else:
                transition_mags = [c_small, 0]
    # Player2 alive, player1 dead
    elif state1 == 0 and state2 != 0:
        # Player2 chooses "far"
        if action2 == "far":
            if succ2 == True:
                transition_mags = [0, g_small]
            else:
                transition_mags = [0, c_big]
        # Player2 chooses "far"
        if action2 == "near":
            if succ2 == True:
                transition_mags = [0, g_small]
            else:
                transition_mags = [0, c_small]
    # Both players dead
    else:
        transition_mags = [0, 0]
    return transition_mags, succ1, succ2

# Define function to get reward
def get_reward(next_state):
    if next_state == 0:
        reward = -1
    else:
        reward = 0
    return reward

# Define function for "step control" (map time on space)
def step(time_point):
    if time_point < days_n - 1:
        return time_point + 1
    else:
        return time_point

# Define function to get action based on epsilon-greedy algorithm
def epsilon_greedy(q_table, weather_i, days_i, state1, state2):
    if np.random.uniform(0, 1) < exploration_rate:
        # Choose a random action
        return ["near", "far"][choice((0, actions_n-1))]
    else:
        # Choose the best action based on Q-value
        return ["near", "far"][np.argmax(q_table[weather_i, :, days_i, state1, state2])]

# Debugging Q-learning algorithm
def debugger(episode,days_i,weather_i,states,action1,action2,p_success1,p_success2,succ1,succ2,gains,next_state1,next_state2,reward1,reward2):
    print("===== EPISODE:", str(episode) + " =====")
    print("time-point:", days_i+1)
    print("weather index:", weather_i)
    print("state1:", states[0], "state2:", states[1])
    print("choice1:", action1 + " forest", "choice2:", action2 + " forest")
    print("p_success1:", p_success1, "p_success2:", p_success2)
    print("success1:", succ1, "success2:", succ2)
    print("transition magnitudes:", gains)
    print("state_next1:", next_state1, "state_next2:", next_state2)
    print("reward1:", reward1,"reward2:", reward2)
    print("Q-value1_update:", q_table1[weather_i, ["near", "far"].index(action1), days_i, states[0], states[1]])
    print("Q-value2_update:", q_table2[weather_i, ["near", "far"].index(action2), days_i, states[1], states[0]])

# Define Q-learning algorithm function
def q_learning(q_table1, q_table2, learning_rate, discount_factor, exploration_rate, episodes):
    # global rewards_per_episode
    rewards_per_episode1 = []
    # Rewards per episode
    total_episode_reward1 = 0
    # Loop over episodes
    for episode in range(episodes):
        # Random weather type
        weather_i = choice((0, weather_n-1))
        # Reset state to starting position
        states = (int(np.random.randint(1, 4)), int(np.random.randint(1, 4)))
        # # Play the game for a mini-block with same exploration rate
        # for blck_i in range(blck_n):
        # Play the game for two time-points
        for days_i in range(days_n):
            # Actions for both players
            action1 = epsilon_greedy(q_table1, weather_i, days_i, states[0], states[1])
            action2 = epsilon_greedy(q_table2, weather_i, days_i, states[1], states[0])
            # Get probabilities for payoffs
            p_success1 = prob(action1, action2, weather_i, states[0], states[1])
            p_success2 = prob(action2, action1, weather_i, states[1], states[0])
            # Calculate payoff
            gains, succ1, succ2 = get_payoffs(states[0], states[1], action1, action2, p_success1, p_success2)
            # Observe final state
            next_state1 = np.clip(states[0]+gains[0], 0, 3)
            next_state2 = np.clip(states[1]+gains[1], 0, 3)
            # Step control
            move = step(days_i)
            # Observe rewards
            reward1 = get_reward(next_state1)
            reward2 = get_reward(next_state2)
            # Update Q-table for current state and action1
            q_table1[weather_i, ["near", "far"].index(action1), days_i, states[0], states[1]] += (learning_rate*(reward1 + discount_factor*np.max(q_table1[0, :, move, next_state1, next_state2]) - q_table1[weather_i, ["near", "far"].index(action1), days_i, states[0], states[1]]))/len(weather_types) + (learning_rate*(reward1 + discount_factor*np.max(q_table1[1, :, move, next_state1, next_state2]) - q_table1[weather_i, ["near", "far"].index(action1), days_i, states[0], states[1]]))/len(weather_types)
            q_table2[weather_i, ["near", "far"].index(action2), days_i, states[1], states[0]] += (learning_rate*(reward2 + discount_factor*np.max(q_table2[0, :, move, next_state2, next_state1]) - q_table2[weather_i, ["near", "far"].index(action2), days_i, states[1], states[0]]))/len(weather_types) + (learning_rate*(reward2 + discount_factor*np.max(q_table2[1, :, move, next_state2, next_state1]) - q_table2[weather_i, ["near", "far"].index(action2), days_i, states[1], states[0]]))/len(weather_types)
            # # Print states, choices and teps
            # debugger(episode,days_i,weather_i,states,action1,action2,p_success1,p_success2,succ1,succ2,gains,next_state1,next_state2,reward1,reward2)
            # Update state
            states = [next_state1, next_state2]
        # # Exploration decay
        # exploration_rate *= exploration_decay
        # Get reward player1
        total_episode_reward1 += reward1
        rewards_per_episode1.append(total_episode_reward1)
    return q_table1, q_table2, rewards_per_episode1

# Run Q-learning and get Q-tables for each player
q_table1, q_table2, rewards_per_episode1 = q_learning(q_table1, q_table2, learning_rate, discount_factor, exploration_rate, episodes)
# Binarized value difference
q_binar1 = q_table1[:][1] - q_table1[:][0]
q_binar2 = q_table2[:][1] - q_table2[:][0]

# =============================================================================
# Evaluate Q-learning
# =============================================================================
print("Reward per thousand episodes")
for i in range(10):
    print(str((i+1)*int(episodes/10)),": Espiode reward: ",\
            rewards_per_episode1[int(episodes/10)*(i+1)-1])

# =============================================================================
# Visualize p_cooperate
# =============================================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
beta = 3.53
value_set = [0, 1]

# Define softmax function for pCooperate
def soft_function(q_diff, beta):
    return 1 / (1 + np.exp(-q_diff*beta))

# Define hardmax policy
def hard_function(q_diff):
    if q_diff < 0:
        return 0
    elif q_diff > 0:
        return 1
    else:
        return 0.5

# Extract data and crop matrices
q_binar1_weather1 = q_binar1[0][0:days_n][:]
q_binar1_weather2 = q_binar1[1][0:days_n][:]
q_binar2_weather1 = q_binar2[0][0:days_n][:]
q_binar2_weather2 = q_binar2[1][0:days_n][:]

# Probabilities to cooperate softmax
p_coop = [soft_function(q_binar1_weather1, beta), soft_function(q_binar2_weather1, beta),
          soft_function(q_binar1_weather2, beta), soft_function(q_binar2_weather2, beta)]
# Policy hard function
policy = [np.vectorize(hard_function)(q_binar1_weather1), np.vectorize(hard_function)(q_binar2_weather1),
          np.vectorize(hard_function)(q_binar1_weather2), np.vectorize(hard_function)(q_binar2_weather2)]

# Define function for activation plot
def plot_pCoop(activation_function):
    # Set publication level params
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'
    ## Canvas
    fig, ax = plt.subplots(figsize=(10, 4), 
                dpi = 600)
    ax = sns.heatmap(np.concatenate(activation_function).T, annot=False, linewidths=.5, linecolor='black')
    ax.invert_yaxis()
    # Set lines for time points
    b, t = plt.ylim()
    ax.vlines(x = states_n, ymin = b, ymax = t, colors = 'black', lw = 3)
    # Custom labels
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[4] = '0'
    labels[5] = '1'
    labels[6] = '2'
    labels[7] = '3'
    ax.set_xticklabels(labels, fontsize = 17)
    ax.set_yticklabels([item.get_text() for item in ax.get_yticklabels()], fontsize = 17)
    plt.show()

# Plotting
target = p_coop
plot_pCoop(target[0])
plot_pCoop(target[1])
plot_pCoop(target[2])
plot_pCoop(target[3])