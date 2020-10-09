""""
Author: Ganga Lingden
This is the Q-learning implement to solve the Taxi problem with optimal policy
"""
# Load OpenAI Gym and other necessary packages
import gym
import random
import time
import numpy as np


# Define Training function
def train(num_of_episodes, num_of_steps, alpha, gamma, Q_reward):
    """
    :param num_of_episodes(int):  Total no. of episode to train
    :param num_of_steps(int):  Number of steps in each Episode
    :param alpha(float):  Learning rate
    :param gamma(flaot):  Future reward discount factor
    :param Q_reward(2d array): Matrix of size(500,6)

    :return:
    Q_reward(2d array): Updated reward Q-table
    """

    for eps in range(num_of_episodes):  # No. of episodes
        state = env.reset()  # set environment

        for step in range(num_of_steps):
            '''
            if np.random.random() < epsilon:
                action = np.random.choice(6) # random action
            else:
                action = np.argmax(Q_reward[state]) # greedy action 
            '''
            action = np.random.choice(6)  # random action
            new_state, reward, done, info = env.step(action)  # next step observation

            q_value = Q_reward[state, action] - alpha * (Q_reward[state, action]) + alpha * (
                    reward + gamma * np.max(Q_reward[new_state]))  # new q-value
            Q_reward[state, action] = q_value  # update q-value
            state = new_state  # update state
        print(f"Episode: {eps}")

    return Q_reward


# Define Test Function
def test(q_table):
    """
    :param q_table(2d array): Updated/trained Q-tables

    :return:
    total_reward(int): total reward
    total_reward(int): total actions/steps
    """

    state = env.reset()  # set environment
    total_reward = 0
    total_action = 0
    for steps in range(50):
        action = np.argmax(Q_reward[state])  # action
        new_state, reward, done, info = env.step(action)  # observe
        state = new_state  # update new state
        total_reward += reward  # reward
        total_action += 1  # count action
        env.render()
        time.sleep(1)

        # Drop passenger successfully
        if done:
            break
    return total_reward, total_action


if __name__ == '__main__':

    # Make environment
    env = gym.make("Taxi-v3")
    print(f"Action Space: {env.action_space}")
    print(f"State Space: {env.observation_space}")

    # Training parameters for Q learning
    alpha = 0.9  # Learning rate
    gamma = 0.9  # Future reward discount factor num_of_episodes = 1000
    epsilon = 0.1
    num_of_episodes = 1000
    num_of_steps = 500  # per each episode

    # Initialize Q-tables for rewards
    Q_reward = -100000 * np.ones((500, 6))
    print(f"Q tables: \n {Q_reward} \n")

    # Train the model
    print(f"Training  start ....")
    start_time = time.time()
    Q_reward_updated = train(num_of_episodes, num_of_steps, alpha, gamma, Q_reward)
    print(f".... Training Done\n")
    end_time = time.time() - start_time
    print(f'Time taken : {end_time} seconds')

    # Test with single episode
    total_reward, total_action = test(Q_reward_updated)
    print(f"Total reward: {total_reward}")
    print(f"Total actions: {total_reward}")

    # Test with 10 Episodes  to find the average actions and rewards
    print(f"\n Testing  with 10 episodes...")
    avg_total_reward = []
    avg_total_actions = []

    for eps in range(10):  # 10 Episodes
        state = env.reset()
        total_reward, total_action = test(Q_reward_updated)

        # append each episode values
        avg_total_reward.append(total_reward)
        avg_total_actions.append(total_action)
    print(f"Average total reward: {np.array(avg_total_reward).mean()}")
    print(f"Average total actions: {np.array(avg_total_actions).mean()}")
