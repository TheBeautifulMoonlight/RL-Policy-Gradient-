import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal,Categorical
from torch import optim
import numpy as np

class Model(nn.Module):
    # 定义网络模型
    def __init__(self,input_n,output_n):
        super().__init__()
        self.first = nn.Linear(input_n,64)#输入层
        self.second = nn.Linear(64,output_n)#输出层
        self.state_head = nn.Linear(64,1)#状态层
        #self.mu_layer = nn.Linear(64,output_n)
        #self.sigma_layer = nn.Linear(128,output_n)
    
    def forward(self,state):
        x = self.first(state)
        x = F.relu(x)
        state_value = self.state_head(x)#输出状态价值
        a = self.second(x)
        action_sorce = F.softmax(a,dim=1)#输出动作的概率
        #mu = F.sigmoid(self.mu_layer(x))
        #sigma = 0.1 #F.softplus(self.sigma_layer(x))
        return action_sorce,state_value

    def get_dist(self,state):
        # 获得动作的分布
        #mu,sigma = self.forward(state)
        #dist = Normal(mu,sigma)
        prob,state_value = self.forward(state)#调用网络
        dist = Categorical(prob)#使用分类分布
        return dist,state_value

class Policy:
    # 策略部分用于输出动作和梯度更新
    def __init__(self,observation_dim,action_dim,lr=0.01,gamma=0.9):
        self.gamma = gamma #折扣累计汇报
        self.net = Model(observation_dim,action_dim)
        self.reward = [] #记录回报
        self.log_prob = [] #记录动作log概率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(),lr=lr) #使用Adam优化器
        self.eps = np.finfo(np.float32).eps.item() #一个非常小的数，为了防止出现分母为0的情况

    def select_action(self,state):
        #从分布中选取一个动作并记录动作的log概率
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.to(self.device)
        dist,state_value = self.net.get_dist(state)
        action = dist.sample()
        prob = dist.log_prob(action)
        self.log_prob.append((prob,state_value))
        action = action.cpu().detach().numpy().flatten()
        return action

    def learn(self):
        # 策略更新
        R = 0
        policy_loss = []
        value_loss = []
        rewards = []
        for r in self.reward[::-1]:
            # 将收益按照折扣累计回报计算
            R = r + self.gamma*R
            rewards.insert(0,R)
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps) #收益归一化
        for (log_prob,state_value), R in zip(self.log_prob, rewards):
            n_R = R - state_value #收益减去基线
            policy_loss.append(-log_prob * n_R) 
            #这就是策略梯度最核心推导的公式，是不是很简单，log概率乘收益作为loss
            value_loss.append(F.smooth_l1_loss(state_value,R)) #状态价值误差
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum() # 策略梯度求和
        value_loss = torch.stack(value_loss).sum() # 状态误差求和
        loss = policy_loss + value_loss # 有点玄学的步骤，将两个loss合一块来更新网络
        loss.backward() # 反向传播
        self.optimizer.step() # 优化参数
        del self.reward[:]
        del self.log_prob[:]
