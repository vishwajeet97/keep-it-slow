#!/bin/bash
set -e
num_procs=$1

declare -A pids=( )

for di in 2 3 5 7 10 15 ;
	do
		for ((i=1; i<=100; i++)); do
		  while (( ${#pids[@]} >= num_procs )); do
		  	# echo ${#pids[@]}
		    wait -n
		    for pid in "${!pids[@]}"; do
		      kill -0 "$pid" &>/dev/null || unset "${pids[$pid]}"
		    done
		  done
		python reinforce-slow.py --seed=$i --di=$di &
		done
	done

for ((i=1; i<=100; i++)); do
  while (( ${#pids[@]} >= num_procs )); do
    wait -n
    for pid in "${!pids[@]}"; do
      kill -0 "$pid" &>/dev/null || unset "${pids[$pid]}"
    done
  done
python reinforce.py --seed=$i
done
