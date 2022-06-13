import gym
import model.Policy as Policy

def Action_adapter(a,max_action):
    return  2*(a-0.5)*max_action

env = gym.make('CartPole-v1')
env.seed(3)
print(env.observation_space)
print(env.action_space.n)
#max_action = env.action_space.high


policy = Policy.Policy(env.observation_space.shape[0],env.action_space.n)
running_reward = 10
for i in range(100):
    state,ep_reward = env.reset(),0
    for t in range(1,1000):
        action = policy.select_action(state)
        #action = Action_adapter(action,max_action)
        state, reward, done, _ = env.step(action[0])
        policy.reward.append(reward)
        #env.render()
        ep_reward += reward
        if done:
            break
    policy.learn()
    running_reward = 0.05 * ep_reward + (1-0.05) * running_reward
    if i%10 == 0:
        print(i,':',running_reward)

state = env.reset()
for t in range(1000):
    action = policy.select_action(state)
    state, reward, done, _ = env.step(action[0])
    policy.reward.append(reward)
    env.render()  
    if done:
        break