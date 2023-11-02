import torch
from models.ac_network import ActorCriticNetwork
import numpy as np
import random
class ActorCriticLoss(torch.nn.Module):
    def __init__(self):
        super(ActorCriticLoss, self).__init__()

    def forward(self, log_prob, delta):
        actor_loss = -log_prob*delta
        critic_loss = delta ** 2
        loss = actor_loss+critic_loss
        return loss.mean()
    
class Agent:
    def __init__(self, obs_space=6, alpha=0.0003, gamma=0.99, epsilon_initial=0.9, epsilon_min=0.1, epsilon_decay=0.995, n_actions=8, eval_mode=False) -> None:
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]
        self.epsilon = epsilon_initial
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = ActorCriticNetwork(n_actions=n_actions, obs_space = obs_space)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=alpha)
        self.loss_fn = ActorCriticLoss()

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if not eval_mode:
            self.model.train()
        else:
            self.model.eval()

        self.model = self.model.to(self.device)
    
    def choose_action(self, observation):
        if random.random() < self.epsilon:
            # Explore: choose a random action
            action = random.choice(self.action_space)
            action = torch.tensor([action])
            self.action = action.to(self.device)
        else:
            # Exploit: choose the action with the highest estimated value
            state = torch.tensor(np.array([observation]), device =self.device, dtype=torch.float32)
            with torch.no_grad():
                _, probs = self.model(state)
            action_probabilities = torch.distributions.Categorical(probs=probs)
            action = action_probabilities.sample()
            self.action = action
        action = action.cpu().numpy()[0]
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return action

    def save_models(self, name="") -> None:
        #print('... saving models ...')
        torch.save(self.model.state_dict(), self.model.checkpoint_file+name)

    def load_models(self, name="") -> None:
        print('... loading model ...')
        state_dict = torch.load(self.model.checkpoint_file+name)
        self.model.load_state_dict(state_dict)
    
    def learn(self, state, reward, state_, done):
        """
            State: current state
            Reward: future reward
            State_: next state
            Done: is it done learning?
        """
        state = torch.tensor(np.array([state]), dtype=torch.float32, device =self.device)
        state_ = torch.tensor(np.array([state_]), dtype=torch.float32, device =self.device )
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

        


