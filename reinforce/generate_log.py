from multiprocessing.dummy import Pool as ThreadPool 
import itertools
import subprocess

processes = set()
def run_reinforce(seed_di):
    seed, di = seed_di
    command = 'python' #'reinforce_slow.py'
    print("sta")
    #subprocess.call([command, 'reinforce-slow.py', '--seed=' + str(seed), '--di=' + str(di)])
    subprocess.call([command, 'reinforce.py', '--seed=' + str(seed)])
    print("end")


seeds = range(4)
dis = [1]
seeds_dis = list(itertools.product(seeds,dis))

pool = ThreadPool(4)
results = pool.map(run_reinforce, seeds_dis)
