# 代码清单1-2　根据指定确定性策略决定动作的智能体
import gym
import numpy as np
import time
env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low, 
     env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))


class BespokeAgent:
    def __init__(self, env):
        pass

    def decide(self, observation): # 决策
        position, velocity = observation


        lb = np.minimum(-0.09 * (position + 0.25) ** 2 + 0.03,
            0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.06

        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action # 返回动作


    def learn(self, *args): # 学习
        pass


agent = BespokeAgent(env)



def play_montecarlo(env, agent:BespokeAgent, render=False, train=False, reward=False):
    episode_reward = 0. # 记录回合奖励，初始化为0
    observation,info = env.reset(seed=0) # 重置游戏环境，开始新回合

    while True:

        if render:
            env.render() # 显示图形界面
            time.sleep(1)
        
        action = agent.decide(observation)
        next_observation, reward, done, _, _ = env.step(action)

        if train:
            agent.learn(observation, action, reward, done)
        if done:
            break

        episode_reward += reward # 收集回合奖励
        observation = next_observation # 更新状态  

    
    return episode_reward
        



# env.seed(0)

episode_reward  = play_montecarlo(env, agent, reward=True)
print("回合奖励 = {}".format(episode_reward))
env.close()


episode_rewards = [play_montecarlo(env, agent, reward=True) for _ in range(100)]
print('平均回合奖励 = {}'.format(np.mean(episode_rewards)))