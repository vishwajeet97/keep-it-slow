#! /bin/bash

array=(2 3 5 7 10 15)

for i in ${array[@]}
do
echo $i
python qlearning-slow.py $i
done

python qlearning.py