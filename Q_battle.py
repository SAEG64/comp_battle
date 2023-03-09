#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:53:53 2023

@author: sergej
Multi-agent Q-learning for a stochastic version of the prisoner's dilemma where
agents have to maintain their energy from decaying to zero by searching for food
(foraging). At each time-point, agents have to decide whether to forage in the 
near or far forest offering different probabilities of success. The environments 
(forests) have different probabilities of success and 
the probabilities resulting from the interaction between the two players' choices 
result in the rules of a prisoner's dilemma.

'Reward shaping' was applied to ensure Q-learning to converge to "real" OP.
"""
# =============================================================================
# Q-learning playing comp 2x2 against it self
# =============================================================================
import numpy as np
from random import choice
from itertools import combinations_with_replacement
from copy import deepcopy
import io
import sys

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

weather_types = [filterPD()[0]]

# Define state matrix size and action matrix size
days_n = 2 # Nr. of timpe-points
states_n = 4  # 0 life points, 1 life point, 2 life points, 3 life points
actions_n = 2  # 0: near forest, 1: far forest
weather_n = len(weather_types)  # 0: weather1, 1: weather2
players_n = 2

# Set hyper parameters Nash Q-learning
episodes = 100
learning_rate = 0.8
discount_factor = 0.9
exploration_rate = 0.2
beta = 4

# Initialize Q-tables for 2 players
q_table1 = np.zeros((weather_n, actions_n, days_n, states_n, states_n))
q_table1[:,:,:,0] = -1
q_table2 = np.zeros((weather_n, actions_n, days_n, states_n, states_n))
q_table2[:,:,:,0] = -1

# Define function to get probability of payoff
def prob(action1, action2, weather_i, state2):
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

# Define function to calculate payoffs
def get_payoffs(state1, state2, action1, action2, p_success1, p_success2, payoffs):
    global transition_mags
    # Probabilistic outcome
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
        # Choose random if equal values
        if q_table[weather_i, 0, days_i, state1, state2] == q_table[weather_i, 1, days_i, state1, state2]:
            return ["near", "far"][choice((0, actions_n-1))]
        # Choose the best action based on Q-value
        else:
            return ["near", "far"][np.argmax(q_table[weather_i, :, days_i, state1, state2])]

# Define function to update Q-value
def q_update(learning_rate, discount_factor, reward, q_current, q_next, state, next_state, days_i, days_n, p_success):
    if days_i != days_n-1 and next_state != 0:
        value_update = learning_rate * (reward * (1-p_success) + discount_factor * q_next - q_current)
        return value_update
    else:
        if state != 0:
            value_update = learning_rate * (reward * (1-p_success) - q_current)
            return value_update
        else:
            value_update = learning_rate * (reward - q_current)
            return value_update

# Debugging Q-learning algorithm
def debugger_both(episode,days_i,weather_i,states,action1,action2,p_success1,p_success2,succ1,succ2,gains,next_state1,next_state2,reward1,reward2,move,state_table,q_current1,q_current2,q_update1,q_update2):
    keys=list(weather_types[weather_i].keys())
    values=list(weather_types[weather_i].values())
    print("=======================")
    print("day_i:", days_i+1, ", weather_i:", weather_i)
    print("================================")
    print("PLAYER1| state:", states[0], ", choice:", action1)
    print("================================")
    print("payoff matrix ind1:", keys[values.index(p_success1)])
    print("p_success1:", round(p_success1,2))
    print("foraging_success1:", succ1)
    print("energy_yield1:", gains[0])
    print("next_state1:", next_state1)
    print("reward1:", reward1)
    print("Q-before1:", q_current1)
    print("Q-update1:", q_update1)
    print("================================")
    print("PLAYER2| state:", states[1], ", choice:", action2)
    print("================================")
    print("payoff matrix ind2:", keys[values.index(p_success2)])
    print("p_success2:", round(p_success2,2))
    print("foraging_success2:", succ2)
    print("energy_yield2:", gains[1])
    print("next_state2:", next_state2)
    print("reward2:", reward2)
    print("Q-before2:", q_current2)
    print("Q-update2:", q_update2)

def debugger_single(episode,days_i,weather_i,states,action,p_success,succ,gains,next_state1,next_state2,reward,move,state_table,q_current,q_update):
    keys=list(weather_types[weather_i].keys())
    values=list(weather_types[weather_i].values())
    print("state:", states[0], ", unchosen:", action)
    print("========================")
    print("payoff matrix ind:", keys[values.index(p_success)])
    print("p_success:", round(p_success,2))
    print("foraging_success:", succ)
    print("energy_yield:", gains[0])
    print("next_state:", next_state1)
    print("reward:", reward)
    print("Q-before:", q_current)
    print("Q-update:", q_update)
    
# Define Q-learning algorithm function
def q_learning(q_table1, q_table2, learning_rate, discount_factor, exploration_rate, episodes):
    # Global rewards_per_episode
    total_rewards1 = []
    total_rewards2 = []
    # Rewards per episode
    summed_rewards1 = 0
    summed_rewards2 = 0
    # Loop over episodes
    for episode in range(episodes):
        print("=============================================")
        print("===== EPISODE:", str(episode+1) + " =====")
        print("=============================================")
        # Random weather type
        weather_i = choice((0, weather_n-1))
        # Reset state to starting position
        states_observed = (int(np.random.randint(1, 4)), int(np.random.randint(1, 4)))
        # Play the game for two time-points
        for days_i in range(days_n):
            # State-action media
            state_table = [deepcopy(q_table1), deepcopy(q_table2)]
            states = states_observed
            # Step control
            move = step(days_i)
            print("=============================================")
            print("===== ACTION-VALUE UPDATES BOTH PLAYERS =====")
            # Actions for both players
            action1 = epsilon_greedy(q_table1, weather_i, days_i, states[0], states[1])
            action2 = epsilon_greedy(q_table2, weather_i, days_i, states[1], states[0])
            # Get probabilities for payoffs
            p_success1 = prob(action1, action2, weather_i, states[1])
            p_success2 = prob(action2, action1, weather_i, states[0])
            # Calculate payoff
            gains, succ1, succ2 = get_payoffs(states[0], states[1], action1, action2, p_success1, p_success2, payoffs)
            # Observe final state
            next_state1 = np.clip(states[0]+gains[0], 0, 3)
            next_state2 = np.clip(states[1]+gains[1], 0, 3)
            # Observe rewards
            reward1 = get_reward(next_state1)
            reward2 = get_reward(next_state2)
            # Track real rewards for evaluation
            reward1_true = reward1
            reward2_true = reward2
            # Get Q-values for current and future state-action pair
            q_curr1 = state_table[0][weather_i,["near", "far"].index(action1),days_i,states[0],states[1]]
            q_next1 = np.max(state_table[0][weather_i,:,move,next_state1,next_state2])
            q_curr2 = state_table[1][weather_i,["near", "far"].index(action2),days_i,states[1],states[0]]
            q_next2 = np.max(state_table[1][weather_i,:,move,next_state2,next_state1])
            # Update Q-values with Nash Q-learning rule
            value_update1 = q_update(learning_rate, discount_factor, reward1, q_curr1, q_next1, states[0], next_state1, days_i, days_n, p_success1)
            value_update2 = q_update(learning_rate, discount_factor, reward2, q_curr2, q_next2, states[1], next_state2, days_i, days_n, p_success2)
            q_table1[weather_i, ["near", "far"].index(action1), days_i, states[0], states[1]] += value_update1
            q_table2[weather_i, ["near", "far"].index(action2), days_i, states[1], states[0]] += value_update2
            # Print states, choices and steps
            debugger_both(episode,days_i,weather_i,states,action1,action2,p_success1,p_success2,succ1,succ2,gains,next_state1,next_state2,reward1,reward2,move,state_table,q_curr1,q_curr2,value_update1,value_update2)
            # Update states observed
            states_observed = [next_state1, next_state2]
            print()
        # Get reward player1
        summed_rewards1 += reward1_true
        summed_rewards2 += reward2_true
        total_rewards1.append(summed_rewards1)
        total_rewards2.append(summed_rewards2)
    return q_table1, q_table2, total_rewards1, total_rewards2

# # Mute stdout out
# text_trap = io.StringIO()
# sys.stdout = text_trap
# Run Q-learning and get Q-tables for each player
q_table1, q_table2, total_rewards1, total_rewards2 = q_learning(q_table1, q_table2, learning_rate, discount_factor, exploration_rate, episodes)
# # Restore stdout function
# sys.stdout = sys.__stdout__
# Binarized value difference
q_binar1 = q_table1[:,1] - q_table1[:,0]
q_binar2 = q_table2[:,1] - q_table2[:,0]

# =============================================================================
# Evaluate Q-learning
# =============================================================================
print("Reward per thousand episodes")
for i in range(10):
    print("========================================")
    print(str((i+1)*int(episodes/10)),": Espiode reward player1: ",\
            total_rewards1[int(episodes/10)*(i+1)-1])
    print(str((i+1)*int(episodes/10)),": Espiode reward player2: ",\
            total_rewards2[int(episodes/10)*(i+1)-1])

# =============================================================================
# Visualize p_cooperate
# =============================================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
beta1 = 4
value_set = [0, 1]

# Define softmax function for pCooperate
def soft_function(q_diff, beta1):
    return 1 / (1 + np.exp(-q_diff*beta1))

# Define hardmax policy
def hard_function(q_diff):
    if q_diff < 0:
        return 0
    elif q_diff > 0:
        return 1
    else:
        return 0.5

# Extract data and crop matrices
q_binar1_day1 = q_binar1[0,0]
q_binar1_day2 = q_binar1[0,1]
q_binar2_day1 = q_binar2[0,0]
q_binar2_day2 = q_binar2[0,1]

# Probabilities to cooperate softmax
p_coop = [np.concatenate([soft_function(q_binar1[0], beta1)]), soft_function(q_binar2[0], beta1)]
# Policy hard function
policy = [np.vectorize(hard_function)(q_binar1[0]), np.vectorize(hard_function)(q_binar2[0])]

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
for i in range(len(target)):
    plot_pCoop(target[i])
