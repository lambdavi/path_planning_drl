"""
    Main file for Path Planning and Obstacle Avoidance DRL
"""
# IMPORTS
import numpy as np
from agents.QAgent import LinearDQN_Agent, ImageDQNAgent
from roverenv import RoverEnv
import wandb

# CFG PARAMETERS
MAX_EPISODES = 1000
N_EPISODES = 1000
TRAIN_MODE = True
LOG_ON = True
LR=0.0001
BS=32
SCHEDULER = False
OBS_SPACE = "image"

if LOG_ON:
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="my-awesome-project",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": LR,
            "epochs": MAX_EPISODES,
            "batch_size": BS,
            "scheduler": SCHEDULER
        })
    
# INITIALIZE ENVIRONMENT & Agent
env = RoverEnv(obs_space=OBS_SPACE, render_mode="not_human")
if OBS_SPACE == "linear":
    agent = LinearDQN_Agent(lr=LR, bs=BS, train=TRAIN_MODE, load_path="model.pt", sched=SCHEDULER)
else:
    agent = ImageDQNAgent(lr=LR, bs=BS, train=TRAIN_MODE, load_path="model.pt", sched=SCHEDULER)
    
rewards_history = []
# INITALIZE TRAIN LOOP
tot_reward = 0
best_reward = -1000

while N_EPISODES>0:
    observation = env.get_obs()
    action = agent.get_action(observation, MAX_EPISODES-N_EPISODES)
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
        avg_rew = np.mean(rewards_history[-50:])

        if SCHEDULER:
            agent.scheduler.step(tot_reward)
        if LOG_ON:
            wandb.log({"Instant_reward": tot_reward, "Avg_reward": avg_rew, "Collected": t_score, "Visited":score})

        print(f"--- Game: {MAX_EPISODES-N_EPISODES} - Instant Reward: {tot_reward} - Avg. Reward: {avg_rew} - Visited: {score} - Collected: {t_score} ---")
        if avg_rew > best_reward and TRAIN_MODE:
            best_reward = avg_rew
            print("Saving model..")
            agent.model.save()

        # END EPISODE
        N_EPISODES-=1
        tot_reward = 0
    
    agent.model.save()
