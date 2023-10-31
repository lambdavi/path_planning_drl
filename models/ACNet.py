import os
from torch import nn, save, load

class ActorCriticNetwork(nn.Module):
    def __init__(self, n_actions, obs_space, fc1_dims=1024, fc2_dims=512, name='actor_critic', chpt_dir='best_model') -> None:
        super(ActorCriticNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')
        self.fc1 = nn.Linear(obs_space, self.fc1_dims)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.val_out = nn.Linear(self.fc2_dims, 1)
        self.pol_out = nn.Linear(self.fc2_dims, self.n_actions)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        v = self.val_out(x)
        p = nn.functional.softmax(self.pol_out(x), dim=1)
        return v, p

    def save(self, name="") -> None:
        #print('... saving models ...')
        save(self.state_dict(), self.checkpoint_file+name)

    