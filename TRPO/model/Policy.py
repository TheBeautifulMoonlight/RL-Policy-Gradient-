from turtle import forward
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from torch import optim

class ActorModel(nn.Module):
    # 定义策略网络模型
    def __init__(self,input_n,output_n):
        super().__init__()
        self.input = nn.Linear(input_n,64)#输入层
        self.mu_layer = nn.Linear(64,output_n)#策略分布均值的输出
        self.sigma_layer = nn.Linear(64,output_n)#策略分布方差的输出
    
    def forward(self,state):
        x = self.input(state)
        x = F.relu(x)
        mu = F.sigmoid(self.mu_layer(x))
        sigma = F.softplus(self.sigma_layer(x))
        return mu,sigma

    def get_dist(self,state):
        # 获得动作的分布
        mu,sigma = self.forward(state)
        dist = Normal(mu,sigma)
        mu,sigma = self.forward(state)#调用网络
        dist = Normal(mu,sigma)#使用分类分布
        return dist

class CirticModel(nn.Module):
    def __init__(self,input_n,output_n):
        super().__init__()
        self.input = nn.Linear(input_n,64)#输入层
        self.state_layer = nn.Linear(64,output_n)#输出层

    def forward(self,state):
        x = self.input(state)
        x = F.relu(x)
        x = self.state_layer(x)
        x = F.relu(x)
        return x

class Policy():
    def __init__(self,input_n,output_n):
        self.critic = CirticModel(input_n,output_n)#状态价值网络用来求优势函数
        self.actor = ActorModel(input_n,output_n)#策略网络
        self.old_actor = ActorModel(input_n,output_n)#备份的策略网络用于更新
        self.critic_optim = optim.Adam(self.net.parameters(),lr=0.01)#状态价值使用Adam优化器