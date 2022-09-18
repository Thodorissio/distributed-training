#!/bin/bash

#Configuration file for slave1
fuser -k 2222/tcp
if [ $1 == "2" ];
then
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22:2222", "192.168.1.1:2222"]}, "task": {"index": 1, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/scripts/worker.py $1 $2
elif [ $1 == "3" ];
then
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22:2222", "192.168.1.1:2222", "192.168.1.3:2222"]}, "task": {"index": 1, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/scripts/worker.py $1 $2
fi