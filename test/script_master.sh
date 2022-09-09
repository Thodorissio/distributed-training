#!/bin/bash

if [ $1 == "1" ]; then
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22:2222"]}, "task": {"index": 0, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/test/worker.py
  echo "Running script for test model only on master vm (who is me)"
elif [ $1 == "2" ];
then
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22:2222", "192.168.1.1:2222"]}, "task": {"index": 0, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/test/worker.py  
  echo "Running script for test model on master vm (who is me) and on vm 1"
elif [ $1 == "3" ];
then
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.2222", "192.168.1.1:2222", "192.168.1.3:2222"]}, "task": {"index": 0, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/test/worker.py
  echo "Running script for test model on all 3 vms , including myself"
fi