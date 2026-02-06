import torch 



class OUNoise():

    def __init__(self, action_space, sigma, mu=0.0, theta=0.15, max_sigma=0.5, min_sigma=0.05, decay_period=100000):
        self.mu = mu 
        self.theta = theta 
        self.sigma = float(sigma) 
        self.max_sigma = max_sigma 
        self.min_sigma = min_sigma
        self.decay_period = decay_period 
        self.action_dim = action_space.shape[0]
        self.low        = torch.tensor(action_space.low).to('cuda') 
        self.high       = torch.tensor(action_space.high).to('cuda') 
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dim) * self.mu 

    def evolve_state(self):
        x = self.state 
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(self.action_dim)
        self.state = x + dx 
        return self.state 

    def get_action(self, action, t):
        ou_state = self.evolve_state().to('cuda')
        # self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t/self.decay_period)
        return ou_state, action, torch.clamp(action + ou_state, self.low, self.high)
    



    
