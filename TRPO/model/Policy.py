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
        self.sigma = 1 #方差先设定为固定值
        #self.sigma_layer = nn.Linear(64,output_n)#策略分布方差的输出
    
    def forward(self,state):
        x = self.input(state)
        x = F.relu(x)
        mu = F.sigmoid(self.mu_layer(x))
        sigma = self.sigma#F.softplus(self.sigma_layer(x))
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
    def __init__(self,input_n,output_n,gamma,lamda):
        self.critic = CirticModel(input_n,output_n)#状态价值网络用来求优势函数
        self.actor = ActorModel(input_n,output_n)#策略网络
        self.old_actor = ActorModel(input_n,output_n)#备份的策略网络用于更新
        self.old_actor.load_state_dict(self.actor.state_dict())#让两个网络参数保持一致
        self.critic_optim = optim.Adam(self.critic.parameters(),lr=0.01)#状态价值使用Adam优化器
        self.critic_loss = nn.MSELoss(reduction='sum')
        self.reward = []#记录奖励回报
        self.log_prob = []#记录动作log概率
        self.state = []#记录状态用来更新critic
        self.action = []#记录动作
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma#折扣累计回报
        self.lamda = lamda#gae参数
        self.eps = np.finfo(np.float32).eps.item()

    def get_gae(self):
        # gae方法计算优势函数
        # 为了减少传递参数在这个函数中进行crtic的更新
        state_tensor = torch.tensor(self.state).to(self.device)#将状态转成tensor
        critic_value = self.critic(state_tensor)#获得状态价值
        reward_tensor = torch.tensor(self.reward).to(self.device)#将reward转成tensor
        returns = torch.zeros_like(reward_tensor)
        advants = torch.zeros_like(reward_tensor)
        #计算gae中间参数
        tmp_return = 0
        pre_value = 0
        tmp_advants = 0
        for t in reversed(range(0, len(reward_tensor))):#从后往前来
            tmp_return = reward_tensor[t] + self.gamma*tmp_return
            tmp_tderror = reward_tensor[t] + self.gamma*pre_value - critic_value[t]
            tmp_advants = tmp_tderror + self.gamma*self.lamda*tmp_advants
            returns[t] = tmp_return
            advants[t] = tmp_advants
            pre_value = critic_value[t]

        advants = (advants - advants.mean()) / (advants.std() + self.eps)#将优势函数归一化
        self.train_crtic(critic_value,returns)#更新状态网络
        return advants

    def train_crtic(self,predict,returns):
        #对状态网络进行更新
        loss = self.critic_loss(predict,returns)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def get_loss(self,advants):
        state_tensor = torch.tensor(self.state).to(self.device)#将状态转成tensor
        logprob_tensor = torch.tensor(self.log_prob).to(self.device)#将对数概率转成tensor
        action_tensor = torch.tensor(self.action).to(self.device)#将动作转成tensor
        dist = self.actor.get_dist(state_tensor)#得到当前策略分布
        new_logprob = dist.log_prob(action_tensor)
        kl = new_logprob-logprob_tensor
        dp_v = torch.exp(new_logprob-logprob_tensor)
        loss = advants.unsqueeze(dim=-1)*dp_v
        return loss.mean(),kl.mean()

    def Fvp(self,kl,p):
        p.detach()
        grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        grads = torch.flatten(grads)

        grads_p = (grads*p).sum()
        hessian_p = torch.autograd.grad(grads_p, self.actor.parameters(), create_graph=True)
        hessian_p = torch.flatten(hessian_p)

        return hessian_p+0.1*p

    def train_actor(self):
        advants = self.get_gae()#计算gae并更新crtic
        #计算损失函数
        actor_loss,actor_kl = self.get_loss(advants)
        #计算l的梯度
        actor_grad = torch.autograd.grad(actor_loss,self.actor.parameters())
        actor_grad = torch.flatten(actor_grad)#将梯度展成1维
        #共轭梯度法得到更新方向
        
        
        

        
