import gymnasium as gym 
from twindelayedddpg.networks import Actor, QFunction 
import torch 
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer 
from tensordict import TensorDict 
import copy 
import numpy as np 
import matplotlib.pyplot as plt 
from gymnasium.wrappers import RecordVideo 
import os 
os.environ["MUJOCO_GL"] = "egl"


mean_episode = []
mean_loss = [] 
mean_loss_va = []
mean_loss_vb = []
mean_actions = []

def softupdate(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


class Agent:
    def __init__(self):
        self.env = gym.make('Hopper-v4')
        self.action_n = self.env.action_space.shape[0]
        self.obs_n = self.env.observation_space.shape[0]
        self.rb = TensorDictReplayBuffer(storage = LazyTensorStorage(500000), batch_size = 256)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(self.obs_n, self.action_n, self.env.action_space).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.qvalue_a = QFunction(self.obs_n + self.action_n).to(self.device)
        self.qvalue_a_target = copy.deepcopy(self.qvalue_a).to(self.device)
        self.qvalue_b = QFunction(self.obs_n + self.action_n).to(self.device)
        self.qvalue_b_target = copy.deepcopy(self.qvalue_b).to(self.device)
        self.training = False

    def collect_init_data(self, episodes):

        obs, _ = self.env.reset()
        total_steps = 0 
        while total_steps < 10000:
            obs, _ = self.env.reset()
            done = False 
            steps = 0 
            while steps < 1000 and done == False:
                steps += 1
                total_steps += 1 
                action = self.env.action_space.sample()
                next_obs, reward, done, _, _ = self.env.step(action)
                td = TensorDict({
                    'obs' : torch.tensor(obs, dtype=torch.float32),
                    'action' : torch.tensor(action, dtype=torch.float32),
                    'reward' : torch.tensor(reward, dtype=torch.float32),
                    'next_obs' : torch.tensor(next_obs, dtype=torch.float32),
                    'done' : torch.tensor(int(done), dtype = torch.float32)
                }, [])

                obs = next_obs 
                self.rb.add(td)
    
    def train(self):

        obs, _ = self.env.reset()
        
        env = self.env
        epr = [] 
        loss_qnet_a = [] 
        loss_qnet_b = []
        loss_anet = []
        actions_net = []
        update_freq = 1
        total_steps_per_epoch = 0 
        while total_steps_per_epoch < 4000:
            done = False 
            obs, _ = env.reset()
            ep_reward = 0 
            max_steps = 1000
            actions = []
            steps = 0 
            
            while not done and steps < max_steps:
                action = self.actor(torch.tensor(obs, dtype=torch.float32).to(self.device))
                actions.append(action.detach().cpu().numpy())
                next_obs, reward, done, _, _ = env.step(action.detach().cpu().numpy())
                ep_reward = ep_reward + reward
                td = TensorDict({
                    'obs' : torch.tensor(obs, dtype=torch.float32),
                    'action' : torch.tensor(action, dtype=torch.float32),
                    'reward' : torch.tensor(reward, dtype=torch.float32),
                    'next_obs' : torch.tensor(int(done), dtype=torch.float32),
                }, [])

                obs = next_obs 
                self.rb.add(td)
                steps += 1
                total_steps_per_epoch+=1


                if total_steps_per_epoch > 10 and self.training == False:
                    self.training = True 

                
                if self.training and total_steps_per_epoch%50 == 0:
                    data = self.rb.sample()
                    obs_ = data['obs'].to(self.device)
                    action_ = data['action'].to(self.device)
                    reward_ = data['reward'].to(self.device)
                    next_obs_ = data['next_obs'].to(self.device)
                    done_ = data['done'].to(self.device)

                    predicted_actions = self.actor_target.forward_pred(next_obs_)

                    target_values_a = self.qvalue_a_target(torch.cat([next_obs_, predicted_actions], dim=1)).squeeze(1)
                    target_values_b = self.qvalue_b_target(torch.cat([next_obs_, predicted_actions], dim=1)).squeeze(1)

                    target_values = torch.min(target_values_a, target_values_b)
                    target_q = reward_ + 0.99*(1-done_)*(target_values)

                    current_q_a = self.qvalue_a(torch.cat([obs_, action_], dim=1)).squeeze(1)
                    current_q_b = self.qvalue_b(torch.cat([obs_, action_], dim=1)).squeeze(1)

                    loss_fn_a = torch.nn.MSELoss()
                    loss_q_a = loss_fn_a(target_q, current_q_a)

                    self.qvalue_a.optimizer.zero_grad()
                    loss_q_a.backward(retain_graph=True)
                    self.qvalue_a.optimizer.step()

                    loss_fn_b = torch.nn.MSELoss()
                    loss_q_b = loss_fn_b(target_q, current_q_b)

                    self.qvalue_b.optimizer.zero_grad()
                    loss_q_b.backward()
                    self.qvalue_b.optimizer.step()

                    if total_steps_per_epoch%2 == 0:
                        actor_pred = self.qvalue_a(torch.cat([obs_, self.actor.forward_pred(obs_)], dim=1))
                        loss_p = -torch.mean(actor_pred,dim=0)

                        self.actor.optimizer.zero_grad()
                        loss_p.backward()
                        self.actor.optimizer.step()

                    loss_qnet_a.append(loss_q_a.detach().cpu().numpy())
                    loss_qnet_b.append(loss_q_b.detach().cpu().numpy())
                    loss_anet.append(loss_p.detach().cpu().numpy())

                    softupdate(self.qvalue_a_target, self.qvalue_a, 0.005)
                    softupdate(self.qvalue_b_target, self.qvalue_b, 0.005)
                    softupdate(self.actor_target, self.actor, 0.005)
            
            epr.append(ep_reward)
            actions_net.append(np.mean(actions))

        mean_episode.append(np.mean(epr))
        mean_loss.append(np.mean(loss_anet))
        mean_loss_va.append(np.mean(loss_qnet_a))
        mean_loss_vb.append(np.mean(loss_qnet_b))
        mean_actions.append(np.mean(actions_net))

    def eval(self, episodes):

        epr_reward = [] 
        for ep in range(0, episodes):
            done = False 
            obs, _ = self.env.reset()
            ep_reward = 0 
            steps = 0
            
            while not done and steps < 999:
                action = self.actor.forward_pred(torch.tensor(obs, dtype=torch.float32).to(self.device))
                next_obs, reward, done, _, _ = self.env.step(action.detach().cpu().numpy())
                ep_reward = ep_reward + reward
                obs = next_obs
                steps+=1 
                
            epr_reward.append(ep_reward)
        

        print("Mean Reward for Current Evaluation")
        print(np.mean(epr_reward))


import sys 
test_sigma = [0.8]
r_seed = [45]
results_a = []
results_b = [] 
results_c = [] 
results_e = []
results_d = []
for seed in r_seed:
    agent = Agent()
    gym.utils.seeding.np_random(seed)
    print("length before adding elements:", len(agent.rb))
    agent.collect_init_data(10)
    print("length after adding elements:", len(agent.rb))
    for training_epochs in range(0, 200):
        agent.train()
    results_a.append(mean_episode)
    results_b.append(mean_loss)
    results_c.append(mean_loss_va)
    results_d.append(mean_actions)
    results_e.append(mean_loss_vb)

    mean_episode = []
    mean_loss = []
    mean_loss_va = []
    mean_loss_vb = []
    mean_actions = []

fig, ax = plt.subplots(1, 5, figsize=(30, 15))
fig.set_dpi(1200)
ax[0].set_title(f"Episode Reward")
ax[1].set_title(f"Actor Loss ")
ax[2].set_title(f"QValue Loss A")
ax[3].set_title(f"QValue Loss B")
ax[4].set_title(f"Mean Actions ")
for series in results_a:
    ax[0].plot(np.arange(len(series)), series, linewidth=2.5)

for series in results_b:
    ax[1].plot(np.arange(len(series)), series, linewidth=2.5)

for series in results_c:
    ax[2].plot(np.arange(len(series)), series, linewidth=2.5)

for series in results_d:
    ax[3].plot(np.arange(len(series)), series, linewidth=2.5)

for series in results_e:
    ax[4].plot(np.arange(len(series)), series, linewidth=2.5)

fig.savefig('exp_1_hooper_td3.png')









