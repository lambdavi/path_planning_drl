"""
    Main file for Path Planning and Obstacle Avoidance DRL
"""
# IMPORTS
import numpy as np
from agents.LQAgent import LinearDQN_Agent
from roverenv import RoverEnv
# GLOBAL PARAMETERS
MAX_EPISODE = 1000
N_EPISODES = 1000

# INITIALIZE ENVIRONMENT & Agent
env = RoverEnv(obs_space="linear")
agent = LinearDQN_Agent()
rewards_history = []
# INITALIZE TRAIN LOOP
tot_reward = 0
# TODO: Add wandb visualiz
while N_EPISODES>0:
    observation = env.get_obs()

    action = agent.get_action(observation, MAX_EPISODE-N_EPISODES)

    reward, done, score, t_score = env.step(action)
    tot_reward+=reward
    observation_ = env.get_obs()
    # Train short memory
    agent.train_short_memory(observation, action, reward, observation_, done)
    # Remember
    agent.remember(observation, action, reward, observation_, done)

    if done:
        env.reset()
        rewards_history.append(tot_reward)
        # Train using a sample from memory
        agent.train_long_memory()
        print(f"--- Game: {MAX_EPISODE-N_EPISODES} - Instant Reward: {tot_reward} - Avg. Reward: {np.mean(rewards_history[:-10])} - Visited: {score} - Collected: {t_score} ---")
        # END EPISODE
        N_EPISODES-=1
        tot_reward = 0
    
    agent.model.save()
