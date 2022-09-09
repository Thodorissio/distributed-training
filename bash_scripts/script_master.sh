#!/bin/bash

if [ $2 == "1" ]; then
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22:2222"]}, "task": {"index": 0, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/python_scripts/$1.py
  echo "Running script for $1 model only on master vm (who is me)"
elif [ $2 == "2" ];
then
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22:2222", "192.168.1.1:2222"]}, "task": {"index": 0, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/python_scripts/$1.py  
  echo "Running script for $1 model on master vm (who is me) and on vm 1"
elif [ $2 == "3" ];
then
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22:2222", "192.168.1.1:2222", "192.168.1.3:2222"]}, "task": {"index": 0, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/python_scripts/$1.py
  echo "Running script for $1 model on all 3 vms , including myself"
fi