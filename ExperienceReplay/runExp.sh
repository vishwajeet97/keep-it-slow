#!/bin/bash

runs=2

envName="Acrobot-v1"
algorithm="ER"
outputFile="data$algorithm$envName.txt"
echo $outputFile
echo $algorithm
for r in `seq 1 $runs`;
do
	echo $r
	cmd="python3 main.py $envName $algorithm -1"
	$cmd
done

di_list=(2 3 5 7 9 13)
# di_list=(1 2 )
algorithm="ER-slow"
echo $algorithm
for d in ${di_list[@]};
do
	echo "di " $d
	for r in `seq 1 $runs`;
	do
		echo $r
		cmd="python3 main.py $envName $algorithm $d"
		$cmd
	done
done

echo "number of algorithms: " 1 + ${#di_list[@]}
echo "total runs: " $runs