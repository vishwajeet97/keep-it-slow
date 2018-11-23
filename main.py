import agent
import gym

env = gym.make('CartPole-v0')
# env = gym.make('Alien-v0')
# env = gym.make('Acrobot-v1')
# ag = agent.BaseAgent(env)
ag = agent.ExperienceReplayAgent(env)
# ag = agent.SlowDownExperienceReplayAgent(env)

ag.set_log_filename()
ag.log()
# ag = agent.FittedQIterationAgent(env)
# ag = agent.ReinforceAgent(env)
ag.learn()