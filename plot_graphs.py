import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage: python3 plot_graphs.py dataFile numAlg totRuns")
		sys.exit(1)

	inputFile = sys.argv[1]
	numAlg = int(sys.argv[2])
	totRuns = int(sys.argv[3])


	di_list=[1, 2, 3, 5, 7, 9, 11, 13]

	first = True
	sum_end_times = []
	episodes = []
	i = 0
	numRuns = 0
	algo = 0
	with open(inputFile, "r") as f:
		for line in f:
			if first:
				if line != "\n":
					tokens = line.strip().split()
					print("tokens", tokens)
					eps = int(tokens[0])
					end_time = float(tokens[1])
					sum_end_times.append(end_time)
					episodes.append(eps)
					i += 1
				else:
					print("run ended\n")
					numRuns += 1
					i = 0
					first = False
					print("len sum_end_times", len(sum_end_times))

			else:
				if line != "\n":
					tokens = line.strip().split()
					print("tokens", tokens)
					eps = int(tokens[0])
					end_time = float(tokens[1])
					sum_end_times[i] += end_time
					i += 1

				else:
					print("run ended\n")
					numRuns += 1
					i = 0


			if numRuns == totRuns:
				# numEpisodes = len(sum_end_times)
				print("numRuns", numRuns)

				avg_end_times = [float(t)/float(totRuns) for t in sum_end_times]

				# plot graph

				plt.xlabel("Episodes")
				plt.ylabel("Average time steps per episode")
				# plt.xticks(np.arange(0, episode_end_times[-1], 1000))
				# plt.title(title)
				label = "Experience Replay"
				if algo > 0:
					label = "DI = %d" % di_list[algo-1]

				plt.plot(episodes, avg_end_times, label=label)

				numRuns = 0
				first = True
				sum_end_times = []
				episodes = []
				algo += 1


	plt.legend(loc='best')
	plt.show()









