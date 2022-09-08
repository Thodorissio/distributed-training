#!/bin/bash

if [ $2 == "1" ]; then
  #TF_CONFIG='{"cluster": {"worker": ["83.212.80.22"]}, "task": {"index": 2, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/python_scripts/$1.py
  echo "Running script for $1 model only on master vm ,not here"
elif [ $2 == "2" ];
then
  #TF_CONFIG='{"cluster": {"worker": ["83.212.80.22", "192.168.1.1"]}, "task": {"index": 2, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/python_scripts/$1.py  
  echo "Running script for $1 model on master vm and on vm 1 ,not here"
elif [ $2 == "3" ];
then
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22", "192.168.1.1", "192.168.1.3"]}, "task": {"index": 2, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/python_scripts/$1.py  
  echo "Running script for $1 model on all 3 vms , including here"
fi