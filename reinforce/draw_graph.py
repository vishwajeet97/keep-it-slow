import matplotlib.pyplot as plt
import numpy as np
import pickle

plot_rewards = []
dis = [1, 2, 3, 5, 7]
seed_len = 4

for di in dis:
	rewards = np.zeros(3500)
	for seed in range(seed_len):
		file_path = 'logs/cart_slow_%d_%d' % (di, seed)
		with open(file_path, 'rb') as f:
			print(file_path)
			data = pickle.load(f)
			reward = data['rewards']
			rewards += reward

	rewards /= seed_len
	plot_rewards.append(rewards)

# sns.set(style="darkgrid")
for index, rewards in enumerate(plot_rewards):
	print(plot_rewards)
	plt.plot(list(range(rewards.shape[0])), rewards, label='di = %d'%dis[index])
plt.legend(loc='lower right')
plt.xlabel('Episodes')
plt.ylabel('Average reward per episode')
plt.show()
