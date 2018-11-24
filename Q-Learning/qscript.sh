#! /bin/bash

array=(2 3 5)

for i in ${array[@]}
do
echo $i
python qlearning-slow.py $i
done
