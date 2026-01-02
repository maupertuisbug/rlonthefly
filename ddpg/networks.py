import torch 




class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.input_features = state_size
        self.output_features = action_size

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.output_features)
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.0003)

    def forward(self, state):
        out = self.net(state)
        error = torch.randn_like(out)
        out = out + error 
        out = torch.clamp(out, min = -1, max = 1)
        return out

class QFunction(torch.nn.Module):
    def __init__(self, state_size_n_action_size):
        super().__init__()
        self.input_features = state_size_n_action_size 

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.0003)

    def forward(self, state_n_action):
        out = self.net(state_n_action)
        return out
