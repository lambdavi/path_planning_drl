from roverenv_v2g_fin import RoverEnvV2
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, A2C, PPO
from agents.actor_critic import Agent
import numpy as np
import wandb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from argparse import ArgumentParser

# ARGUMENT PARSER
parser = ArgumentParser()
parser.add_argument('--algo', type=str, default='dqn')
parser.add_argument('--obs', type=str, default='image')
parser.add_argument('--sb', action='store_true')
args = parser.parse_args()

OBS_TYPE = args.obs
env = RoverEnvV2(obs_type=OBS_TYPE)
agent = Agent(3,n_actions=8)
N_GAMES = 1000
load_checkpoint = False
score_history = []
LOG_ON = True

print(args)
if args.sb:
    log_dir = "tmp/"
    env = Monitor(env, log_dir)
    policy = "CnnPolicy" if OBS_TYPE == "image" else "MlpPolicy"
    print(policy)
    if args.algo == "dqn":
        model = DQN(env=env, policy=policy, policy_kwargs=dict(normalize_images=False), tensorboard_log=log_dir, verbose=1, buffer_size=100)
    elif args.algo == "a2c":
        model = A2C(env=env, policy=policy, policy_kwargs=dict(normalize_images=False), tensorboard_log=log_dir, verbose=1)
    else:
        model = PPO(env=env, policy=policy, policy_kwargs=dict(normalize_images=False), tensorboard_log=log_dir, verbose=1)
        
    # Train the agent
    if LOG_ON:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project="a2c-test",
            monitor_gym=True,
            sync_tensorboard=True,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": agent.lr,
                "visited": env.get_wrapper_attr('cells_visited'),
                "collected": env.get_wrapper_attr('targets_collected'),
                "epochs": N_GAMES,
            })
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, save_path=log_dir, name_prefix="ddq_"
    )
    model.learn(
        total_timesteps=1000000 if OBS_TYPE=="linear" else 50000,
        callback=[
            checkpoint_callback,
            WandbCallback(
                gradient_save_freq=10000,
                model_save_path=f"models/{run.id}",
                model_save_freq=10000,
                log="all",
                verbose=2,
            ),
        ],
        progress_bar=True
    )
else:
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