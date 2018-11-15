import agent
import gym

env = gym.make('CartPole-v0')
# env = gym.make('Acrobot-v1')
ag = agent.ExperienceReplayAgent(env)
ag.learn()