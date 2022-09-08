#!/bin/bash

if [ $2 == "1" ]; then
  #activate virtual env 
  #change to correct paths
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22"]}, "task": {"index": 0, "type": "worker"}}' python ~/distirbuted-training/python_scripts_$1.py
  echo "Running script for $1 model only on master vm (who is me)"
elif [ $1 == "2" ];
then
  #activate virtual env 
  #change to correct paths
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22", "192.168.1.1"]}, "task": {"index": 0, "type": "worker"}}' python ~/distirbuted-training/python_scripts_$1.py  
  echo "Running script for $1 model on master vm (who is me) and on vm 1"
elif [ $1 == "3" ];
then
  #activate virtual env 
  #change to correct paths
  TF_CONFIG='{"cluster": {"worker": ["83.212.80.22", "192.168.1.1", "192.168.1.3"]}, "task": {"index": 0, "type": "worker"}}' python ~/distirbuted-training/python_scripts_$1.py  
  echo "Running script for $1 model on all 3 vms , including myself"
fi