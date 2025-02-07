import gym
import numpy as np
from collections import defaultdict


env = gym.make('CliffWalking-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('状态数量 = {}, 动作数量 = {}'.format(env.nS, env.nA))
print('地图大小 = {}'.format(env.shape))




def play_once(env, policy):
    '''
    用给定的策略运行一个回合
    env: 游戏环境对象
    policy: 策略
    '''
    total_reward = 0
    state = env.reset() # 初始位置
    while True:
        '''
        作用是将一个一维索引 state 转换为多维索引 loc，其中 env.shape 是环境的形状。

            在强化学习中，环境的状态通常是用一个整数来表示的，这个整数是状态空间中的一个索引。然而，在某些情况下，我们可能需要将这个一维索引转换为多维索引，以便更好地理解状态在环境中的位置。

            np.unravel_index 是 NumPy 库中的一个函数，它接受两个参数：一个一维索引和一个形状元组。它返回一个元组，其中包含了对应于给定形状的多维索引。

            例如，如果 state 是 10，env.shape 是 (4, 12)，那么 np.unravel_index(state, env.shape) 将返回 (0, 10)，表示状态 10 在环境中的位置是第 0 行第 10 列。
        '''
        # state: 0-46
        loc = np.unravel_index(state, env.shape)
        print('状态 = {}, 位置 = {}'.format(state, loc), end=' ')
        # 据概率分布 p 从动作空间中随机选择一个动作。
        action = np.random.choice(env.nA, p=policy[state]) 
        state, reward, done, _ = env.step(action)
        print('动作 = {}, 奖励 = {}'.format(action, reward))
        total_reward += reward
        if done:
            break
    return total_reward


# 0表示向上，1表示向右，2表示向下，3表示向左
# env.shape =  (4, 12)
actions = np.ones(env.shape, dtype=int)
actions[-1, :] = 0
actions[:, -1] = 2
# 将动作数组转为最优策略
'''
np.eye(4) 创建一个4x4的单位矩阵，其中对角线上的元素为1，其余元素为0。

actions.reshape(-1) 将 actions 数组转换为一维数组，shape = (48,)
然后使用这个一维数组作为索引从单位矩阵中选择相应的行，
从而得到最优策略。 pi.shape = (48, 4)

最优策略是一个二维数组，其中每一行表示一个状态下的动作概率分布。 简单起见，每个概率分布都是一个one-hot向量，
'''
optimal_policy = np.eye(4)[actions.reshape(-1)] # shape = (48, 4)


total_reward = play_once(env, optimal_policy)
print('总奖励 = {}'.format(total_reward))