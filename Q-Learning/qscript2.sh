#! /bin/bash

array=(7 10 15)

for i in ${array[@]}
do
echo $i
python qlearning-slow.py $i
done

python qlearning.py