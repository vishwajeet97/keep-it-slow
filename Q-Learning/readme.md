#q-learning

Run the following commands to test the algorithm on opengym standard control tasks (default MountainCar-v0)

`python qlearning.py`

`python qlearning.py <di>`

The combined thing can be run for different values of d using

`./qscript.sh`

By default these run for 9000 episodes and average over 6 runs. These can be changed by changing the corresponding values in constants.py. The environment is rendered on completion of each run.
