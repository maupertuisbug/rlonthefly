import gymnasium as gym
from ddpg.networks import Actor, QFunction
import torch 
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from tensordict import TensorDict
import copy
import numpy as np 
import matplotlib.pyplot as plt

mean_episode = []
mean_loss = [] 
mean_loss_v = []
mean_actions = []
def softupdate(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )

class Agent:
    def __init__(self, run_type):
        self.env =  gym.make("MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.1)
        self.action_n = self.env.action_space.shape[0]
        self.obs_n = self.env.observation_space.shape[0]
        self.rb = TensorDictReplayBuffer(storage=LazyTensorStorage(300000), batch_size=256)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(self.obs_n, self.action_n, run_type, self.env.action_space).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.qvalue = QFunction(self.obs_n+self.action_n, run_type).to(self.device)
        self.qvalue_target = copy.deepcopy(self.qvalue).to(self.device)

    def collect_init_data(self, episodes):

        obs, _ = self.env.reset()
        for ep in range(0, episodes):
            obs, _ = self.env.reset()
            done = False
            steps = 0
            while steps < 999 and done == False:
                steps += 1
                action = self.env.action_space.sample()
                next_obs, reward, done, _, _ = self.env.step(action)
                td = TensorDict({
                    'obs' : torch.tensor(obs), 
                    'action' : torch.tensor(action), 
                    'reward' : torch.tensor(reward), 
                    'next_obs' : torch.tensor(next_obs),
                    'done' : torch.tensor(int(done)),
                }, [])
                obs = next_obs
                self.rb.add(td)

    def train(self, noise_type):

        ## interact and collect and eval 
        eval_ep = 100
        obs, _ = self.env.reset()
        epr = []
        loss_qnet = [] 
        loss_anet = []
        actions_net = []
        update_freq = 1
        for ep in range(0, eval_ep):
            done = False 
            obs, _  = self.env.reset()
            ep_reward = 0
            max_steps = 999
            actions = []
            steps = 0
            self.actor.noise.reset()
            while not done and steps < max_steps:
                action = self.actor(torch.tensor(obs).to(self.device), noise_type, steps)
                actions.append(action.detach().cpu().numpy())
                next_obs, reward, done, _, _ = self.env.step(action.detach().cpu().numpy())
                ep_reward = ep_reward + reward
                td = TensorDict({
                    'obs' : torch.tensor(obs), 
                    'action' : torch.tensor(action), 
                    'reward' : torch.tensor(reward), 
                    'next_obs' : torch.tensor(next_obs),
                    'done' : torch.tensor(int(done)),
                }, [])
                obs = next_obs
                self.rb.add(td)
                steps+=1
                ## Training
                data = self.rb.sample()
                obs_ = data['obs'].to(self.device)
                action_ = data['action'].to(self.device)
                reward_ = data['reward'].to(self.device)
                next_obs_ = data['next_obs'].to(self.device)
                done_ = data['done'].to(self.device)

                ## compute targets 
                predicted_actions = self.actor_target.forward_pred(next_obs_)
                target_values = self.qvalue_target(torch.cat([next_obs_, predicted_actions], dim=1)).squeeze(1)

                target_q = reward_ + 0.99*(1-done_)*(target_values)
                current_q = self.qvalue(torch.cat([obs_, action_], dim=1)).squeeze(1)

                loss_fn = torch.nn.MSELoss()
                loss_q = loss_fn(target_q, current_q)

                self.qvalue.optimizer.zero_grad()
                loss_q.backward()
                self.qvalue.optimizer.step()

                actor_pred = self.qvalue(torch.cat([obs_, self.actor.forward_pred(obs_)], dim=1))
                loss_p = -torch.mean(actor_pred,dim=0)

                self.actor.optimizer.zero_grad()
                loss_p.backward()
                self.actor.optimizer.step()

                loss_qnet.append(loss_q.detach().cpu().numpy())
                loss_anet.append(loss_p.detach().cpu().numpy())

                if steps % update_freq == 0:
                    softupdate(self.qvalue_target, self.qvalue, 0.005)
                    softupdate(self.actor_target, self.actor, 0.005)
            
            epr.append(ep_reward)
            actions_net.append(np.mean(actions))

        mean_episode.append(np.mean(epr))
        mean_loss.append(np.mean(loss_anet))
        mean_loss_v.append(np.mean(loss_qnet))
        mean_actions.append(np.mean(actions_net))


import sys 
run_type = sys.argv[1]
noise_type = sys.argv[2]
agent = Agent(int(run_type))
agent.collect_init_data(100)
for training_steps in range(0, 1000):
    agent.train(int(noise_type))
    fig, ax = plt.subplots(1, 4, figsize=(30, 15))
    ax[0].plot(np.arange(len(mean_episode)), mean_episode)
    ax[0].set_title(f"Episode Reward "+str(training_steps))

    ax[1].plot(np.arange(len(mean_loss)), mean_loss)
    ax[1].set_title(f"Actor Loss "+str(training_steps))

    ax[2].plot(np.arange(len(mean_loss_v)), mean_loss_v)
    ax[2].set_title(f"QValue Loss "+str(training_steps))

    ax[3].plot(np.arange(len(mean_actions)),  mean_actions)
    ax[3].set_title(f"Mean Actions "+str(training_steps))

    fig.savefig('ddpg_'+str(run_type)+'_'+str(noise_type)+'.png')

