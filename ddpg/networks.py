import torch 
from ou_noise import OUNoise
import gymnasium as gym



class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size, type, sigma, action_space):
        super().__init__()
        self.input_features = state_size
        self.output_features = action_size
        self.action_space = action_space
        self.noise = OUNoise(action_space, sigma)

        if type == 0:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.input_features, 400),
                torch.nn.ReLU(),
                torch.nn.Linear(400, 300),
                torch.nn.ReLU(),
                torch.nn.Linear(300, self.output_features)
            )
        
        else :
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.input_features, 400),
                torch.nn.BatchNorm1d(400),
                torch.nn.ReLU(),
                torch.nn.Linear(400, 300),
                torch.nn.BatchNorm1d(300),
                torch.nn.ReLU(),
                torch.nn.Linear(300, self.output_features)
            )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.001)

    def forward(self, state, type, step):
        out = self.net(state)
        if type == 0:
            error = torch.normal(mean=0, std=0.1, size=out.shape).to('cuda')
            out = out + error 
            out = torch.clamp(out, torch.tensor(self.action_space.low).to('cuda'), torch.tensor(self.action_space.high).to('cuda'))
        else :
            oustate, action, out = self.noise.get_action(out, step)
            out = out.to('cuda')
        return oustate, action, out

    def forward_pred(self, state):
        out = self.net(state)
        out = torch.clamp(out, torch.tensor(self.action_space.low).to('cuda'), torch.tensor(self.action_space.high).to('cuda'))
        return out

class QFunction(torch.nn.Module):
    def __init__(self, state_size_n_action_size, type):
        super().__init__()
        self.input_features = state_size_n_action_size 

        if type == 0:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.input_features, 400),
                torch.nn.ReLU(),
                torch.nn.Linear(400, 300),
                torch.nn.ReLU(),
                torch.nn.Linear(300, 1)
            )
        
        else :
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.input_features, 400),
                torch.nn.BatchNorm1d(400),
                torch.nn.ReLU(),
                torch.nn.Linear(400, 300),
                torch.nn.BatchNorm1d(300),
                torch.nn.ReLU(),
                torch.nn.Linear(300, 1)
            )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.001)

    def forward(self, state_n_action):
        out = self.net(state_n_action)
        return out



class Test:
    def __init__(self):
        env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.1)
        action_space = env.action_space
        self.actor = Actor(5, 2, 1, action_space).to('cuda')
        self.qfunction = QFunction(8, 1).to('cuda')


    def test_batch(self):
        input_a = torch.ones(size = [5]).to('cuda')
        input_b = torch.ones(size = [2]).to('cuda')
        self.actor.eval()
        input_a = input_a.unsqueeze(0)
        value = self.actor(input_a, 1, 1)
        print(value.shape)
        input_c = torch.ones(size = [512, 8]).to('cuda')
        value = self.qfunction(input_c)
        print(value.shape)
