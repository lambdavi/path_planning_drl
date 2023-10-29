"""
    Main file for Path Planning and Obstacle Avoidance DRL
"""
# IMPORTS
import numpy as np
from agents.LQAgent import LinearDQN_Agent
from roverenv import RoverEnv
import wandb
# GLOBAL PARAMETERS
MAX_EPISODE = 5000
N_EPISODES = 5000
TRAIN_MODE = True

wandb.login()

LR=0.0001

run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": LR,
        "epochs": MAX_EPISODE,
    })
# INITIALIZE ENVIRONMENT & Agent
env = RoverEnv(obs_space="linear", render_mode="not_human")
agent = LinearDQN_Agent(lr=LR, train=TRAIN_MODE)
rewards_history = []
# INITALIZE TRAIN LOOP
tot_reward = 0
best_reward = -1000
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
        avg_rew = np.mean(rewards_history[:-10])
        wandb.log({"Instant_reward": tot_reward, "Avg_reward": avg_rew, "Collected": t_score, "Visited":score})

        print(f"--- Game: {MAX_EPISODE-N_EPISODES} - Instant Reward: {tot_reward} - Avg. Reward: {avg_rew} - Visited: {score} - Collected: {t_score} ---")
        if tot_reward > avg_rew and TRAIN_MODE:
            best_reward = tot_reward
            print("Saving model..")
            agent.model.save()

        # END EPISODE
        N_EPISODES-=1
        tot_reward = 0
    
    agent.model.save()
