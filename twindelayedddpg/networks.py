import torch 
import gymnasium as gym 



class Actor(torch.nn.Module):

    def __init__(self, state_size, action_size, action_space):
        super().__init__()
        self.input_features = state_size 
        self.output_features = action_size 
        self.action_space = action_space 

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_features, 400),
            torch.nn.ReLU(), 
            torch.nn.Linear(400, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, self.output_features)
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.001)

    def forward(self, state):
        out = self.net(state)
        error = torch.normal(mean = 0, std = 0.1, size = out.shape).to('cuda')
        out = out + error
        out = torch.clamp(out, torch.tensor(self.action_space.low).to('cuda'), torch.tensor(self.action_space.high).to('cuda'))
        return out

    def forward_pred(self, state):
        out = self.net(state)
        error = torch.normal(mean = 0, std = 0.2, size = out.shape).to('cuda')
        error = torch.clamp(error, -0.4, 0.4)
        out = out + error
        out = torch.clamp(out, torch.tensor(self.action_space.low).to('cuda'), torch.tensor(self.action_space.high).to('cuda'))
        return out


class QFunction(torch.nn.Module):
    def __init__(self, state_size_n_action_size):
        super().__init__()
        self.input_features = state_size_n_action_size 

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_features, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 1)
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.001)

    def forward(self, state_n_action):
        out = self.net(state_n_action)
        return out

    