# 强化学习之策略梯度系列方法(pytorch复现)
这段时间需要用到基于策略梯度的一种强化学习方法，无奈对于方法的核心理解不是很通透，所以一狠心决定从策略梯度开始将这一脉的所有方法都理一遍，加深理解，分享出来与大家共同学习。（也算挖了个大坑，这一段时间会专注把这个坑填上。）

本栏目持续更新一下算法的pyrotch复现
1. 策略梯度算法(PG)
2. 置信域搜索策略(TRPO)---(施工中)
3. 近端策略优化(PPO)---(施工中)
4. 深度确定性策略梯度算法(DDPG)---(施工中)
5. 引导策略搜索(GPS)---(施工中)

使用方法：
在每个方法文件夹下运行
```
python train.py
```
即可使用该算法，并且将策略实现放在model文件夹下。算法的讲解可以在知乎找到，链接地址如下：

[1.策略梯度算法](https://zhuanlan.zhihu.com/p/528037507)