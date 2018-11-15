import agent
import gym

env = gym.make('CartPole-v0')
ag = agent.ExperienceReplayAgent(env)
ag.learn()