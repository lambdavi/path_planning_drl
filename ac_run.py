from roverenv_v2g_p import RoverEnvV2
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from agents.actor_critic import Agent
import numpy as np
import wandb
env = RoverEnvV2()
agent = Agent(3,n_actions=8)
N_GAMES = 1000
load_checkpoint = False
score_history = []
LOG_ON = True
model = DQN(env=env, policy="CnnPolicy", buffer_size=100)
model.learn(100000, progress_bar=True)
exit(1)
if LOG_ON:
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="a2c-test",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": agent.lr,
            "epochs": N_GAMES,
        })
for i in range(N_GAMES):
    observation = env.reset()[0]
    score=0
    done = False
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, _, _ = env.step(action)
        score += reward
        if not load_checkpoint:
            agent.learn(observation, reward, observation_, done)
        observation = observation_
        #env.render()
    score_history.append(score)
    avg_rew = np.mean(score_history[-20:])
    print(f"Episode: {i}, Instant_reward: {score}, Avg Reward: {avg_rew}, Visited: {env.cells_visited}, Targets Collected: {env.targets_collected}")
    if LOG_ON:
            wandb.log({"Instant_reward": score, "Avg_reward": avg_rew, "Collected": env.targets_collected, "Visited":env.cells_visited})
    # Render the game

#env.close()