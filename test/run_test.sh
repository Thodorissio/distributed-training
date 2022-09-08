#!/bin/bash

echo "Training test model in $1 distirbuted machines"


/home/user/distirbuted-training/test/script_master.sh $1  &
ssh user@192.168.1.1 'bash -s' < /home/user/distirbuted-training/test/script_1.sh $1  & #tsekare ta paths
ssh user@192.168.1.3 'bash -s' < /home/user/distirbuted-training/test/script_3.sh $1  &

wait #1
echo "All 3 complete"