import torch 
from ou_noise import OUNoise



class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size, type, action_space):
        super().__init__()
        self.input_features = state_size
        self.output_features = action_size
        self.action_space = action_space
        self.noise = OUNoise(action_space)

        if type == 0:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.input_features, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, self.output_features)
            )
        
        else :
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.input_features, 256),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, self.output_features)
            )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.0003)

    def forward(self, state, type, step):
        out = self.net(state)
        if type == 0:
            error = torch.normal(mean=0, std=0.1, size=out.shape).to('cuda')
            out = out + error 
            out = torch.clamp(out, torch.tensor(self.action_space.low).to('cuda'), torch.tensor(self.action_space.high).to('cuda'))
        else :
            out = self.noise.get_action(out, step).to('cuda')
        return out

    def forward_pred(self, state):
        out = self.net(state)
        return out

class QFunction(torch.nn.Module):
    def __init__(self, state_size_n_action_size, type):
        super().__init__()
        self.input_features = state_size_n_action_size 

        if type == 0:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.input_features, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1)
            )
        
        else :
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.input_features, 256),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1)
            )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.003)

    def forward(self, state_n_action):
        out = self.net(state_n_action)
        return out
