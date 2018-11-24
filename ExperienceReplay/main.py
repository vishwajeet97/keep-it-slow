import agent
import gym
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":
	
	env_name = sys.argv[1]
	algorithm = sys.argv[2]
	di = int(sys.argv[3])

	env = gym.make(env_name)

	ag = agent.ExperienceReplayAgent(env)

	if algorithm == "ER-slow":
		ag = agent.SlowDownExperienceReplayAgent(env, decision_interval=di)
	elif algorithm == "QI":
		ag = FittedQIterationAgent(env)
	elif algorithm == "QI-slow":
		pass

	# env = gym.make('CartPole-v0')
	# env = gym.make('Alien-v0')
	# env = gym.make('Acrobot-v1')
	# ag = agent.BaseAgent(env)
	# ag = agent.ExperienceReplayAgent(env)
	# ag = agent.SlowDownExperienceReplayAgent(env)
	# ag = agent.FittedQIterationAgent(env)
	# ag = agent.ReinforceAgent(env)
	ag.set_log_filename()
	ag.log()
	ag.learn()

