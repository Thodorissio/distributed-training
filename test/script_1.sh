#!/bin/bash

if [ $1 == "1" ]; then
  #TF_CONFIG='{"cluster": {"worker": ["83.212.80.22"]}, "task": {"index": 1, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/test/worker.py
  echo "Running script for test model only on master vm , not here"
elif [ $1 == "2" ];
then
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22:22", "192.168.1.1"]}, "task": {"index": 1, "type": "worker"}}'/home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/test/worker.py  
  echo "Running script for test model on master vm and on vm 1 (who is me)"
elif [ $1 == "3" ];
then
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22:22", "192.168.1.1", "192.168.1.3"]}, "task": {"index": 1, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/test/worker.py  
  echo "Running script for test model on all 3 vms , including here"
fi