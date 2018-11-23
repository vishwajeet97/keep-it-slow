import matplotlib.pyplot as plt
import numpy as np
import pickle

plot_rewards = []
dis = [2, 3, 5, 7, 10, 15]
for di in dis:
	rewards = np.zeros(4000)
	for seed in range(0,5):
		file_path = 'logs/cart_slow_%d_%d'
		with open(file_path%(di,seed), 'r') as f:
			data = pickle.load(f)
			reward = data['rewards']
			rewards += reward

	rewards /= 5
	plot_rewards.append(rewards)

# plot_rewards.append(np.arange(100))
# plot_rewards.append(np.arange(100)+50)
# plot_rewards.append(np.arange(100)+10)
# plot_rewards.append(np.arange(100)+80)

# sns.set(style="darkgrid")
for index, rewards in enumerate(plot_rewards):
	print(plot_rewards)
	plt.plot(list(range(rewards.shape[0])), rewards, label='di = %d'%dis[index])
plt.legend(loc='lower right')
plt.xlabel('Episodes')
plt.ylabel('Average reward per episode')
plt.show()