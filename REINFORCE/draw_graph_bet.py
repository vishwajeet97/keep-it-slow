import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# sns.set(style="darkgrid")

# # Load an example dataset with long-form data
# fmri = sns.load_dataset("fmri")

# import pdb; pdb.set_trace()

# # Plot the responses for different events and regions
# sns.lineplot(x="timepoint", y="signal",
#              hue="region", style="event",
#              data=fmri)

plot_rewards = []
seed_len = 4
dis = [1, 2, 3, 5, 7]
# rewards = np.zeros((len(dis), seed_len, 3500))
rewards = []
for i, di in enumerate(dis):
	for seed in range(seed_len):
		file_path = 'logs/cart_slow_%d_%d' % (di, seed)
		with open(file_path, 'rb') as f:
			print(file_path)
			data = pickle.load(f)
			rewards.extend( [ [ep, r, di, seed] for ep, r in enumerate(data['rewards'].tolist())])


# import pdb; pdb.set_trace()
dat = pd.DataFrame(data=rewards, columns=['Episodes', 'Avg Reward', 'DI', 'seed'])
sns_plt = sns.lineplot(x="Episodes", y="Avg Reward",
             hue="DI", data=dat)
# sns_plt.savefig('DI_REINFORCE.png')
plt.show()