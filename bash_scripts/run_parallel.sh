#!/bin/bash

echo "Training $1 model in $2 distirbuted machines"


./script_master.sh $1 $2 &
ssh user@192.168.1.1 'bash -s' < ~/distirbuted-training/bash_scripts/script_1.sh $1 $2 & #tsekare ta paths
ssh user@192.168.1.3 'bash -s' < ~/distirbuted-training/bash_scripts/script_3.sh $1 $2 &

wait
echo "All 3 complete"