import torch
from models.ACNet import ActorCriticNetwork
import numpy as np
import random
from agents.general import GeneralAgent

class ActorCriticLoss(torch.nn.Module):
    def __init__(self):
        super(ActorCriticLoss, self).__init__()

    def forward(self, log_prob, delta):
        self.actor_loss = -log_prob*delta
        self.critic_loss = delta ** 2
        self.loss = self.actor_loss+self.critic_loss
        return self.loss.mean()
    
class ACAgent(GeneralAgent):
    def __init__(self, obs_space_dim=50, lr=0.0003, gamma=0.99, epsilon=0.1, n_actions=8, eval_mode=False) -> None:
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.epsilon = epsilon
        self.model = ActorCriticNetwork(n_actions=n_actions, obs_space = obs_space_dim)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.loss_fn = ActorCriticLoss()
        self.eval_mode = eval_mode
        if not eval_mode:
            self.model.train()
        else:
            self.model.eval()
    
    def get_action(self, observation, episode):
        self.eps = 200 - episode # random function
        if random.randint(0, 400) < self.eps and not self.eval_mode:
            # increasing n_games we don't get moves anymore
            action = random.randint(0, self.n_actions-1)
            self.action=torch.tensor(action)
        else:
            # Exploit: choose the action with the highest estimated value
            state = torch.tensor(np.array([observation]))
            with torch.no_grad():
                _, probs = self.model(state)
            action_probabilities = torch.distributions.Categorical(probs=probs)
            action = action_probabilities.sample()
            self.action = action
            action = action.numpy()[0]
        return action
    
    def load(self, name="") -> None:
        print('... loading model ...')
        state_dict = torch.load(self.model.checkpoint_file+name)
        self.model.load_state_dict(state_dict)
    def get_loss(self):
        return self.loss_fn.actor_loss, self.loss_fn.critic_loss, self.loss_fn.loss
    def train_step(self, state, reward, state_, done):
        """
            State: current state
            Reward: future reward
            State_: next state
            Done: is it done learning?
        """
        state = torch.tensor(np.array([state]), dtype=torch.float32)
        state_ = torch.tensor(np.array([state_]), dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        
        state_value, probs = self.model(state)
        state_value_, _ = self.model(state_)
        state_value = torch.squeeze(state_value)
        state_value_ = torch.squeeze(state_value_)

        action_probs = torch.distributions.Categorical(probs = probs)
        log_prob = action_probs.log_prob(self.action)
        
        # Temporal difference delta (0 future value if done)
        delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
        #actor_loss = -log_prob*delta
        #critic_loss = delta ** 2
        #total_loss = torch.tensor(actor_loss+critic_loss, requires_grad=True)
        total_loss = self.loss_fn(log_prob, delta)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        


