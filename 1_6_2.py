import gym
from gym import envs


env = gym.make('CartPole-v0')
env_specs = envs.registry.all()

env_ids = [spec.id for spec in env_specs] # 
print(env_ids)