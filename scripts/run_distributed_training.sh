#!/bin/sh

#Usage
usage()
{
    echo 'Usage : ./run_distributed_training -n <nodes> -d <dataset>'
    echo '        nodes: number of nodes for the distributed training'
    echo '        dataset: ml dataset and its corresponding model we wish to train'
    exit
}

if [ $# -ne 4 ]
then
    usage
fi

#Read Arguments
while [ "$1" != "" ]; do
case $1 in
        -n )           shift
                       NODES=$1
                       ;;
        -d )           shift
                       DATASET=$1
                       ;;
    esac
    shift
done

#Check if nodes in range
if [ $NODES -ge 4 ] || [ $NODES -le 0 ]
then
    echo 'Nodes valid values between 1 - 3'
    exit
fi

#Run distributed training
fuser -k 2222/tcp #empty port 2222

if [ "$NODES" -eq 1 ]; then
    echo 'Training in 1 node'
    TF_CONFIG='{"cluster": {"worker": ["83.212.80.22:2222"]}, "task": {"index": 0, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/scripts/worker.py $NODES $DATASET master
elif [ "$NODES" -eq 2 ]; then
    echo 'Training in 2 nodes'
    TF_CONFIG='{"cluster": {"worker": ["83.212.80.22:2222", "192.168.1.1:2222"]}, "task": {"index": 0, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/scripts/worker.py $NODES $DATASET master &
    ssh slave1 'bash -s' < first_slave_config.sh $NODES $DATASET
elif [ "$NODES" -eq 3 ]; then
    echo 'Training in 3 nodes'
    TF_CONFIG='{"cluster": {"worker": ["83.212.80.22:2222", "192.168.1.1:2222", "192.168.1.3:2222"]}, "task": {"index": 0, "type": "worker"}}' /home/user/miniconda3/envs/distributed_training/bin/python3 /home/user/distributed-training/scripts/worker.py $NODES $DATASET master &
    ssh slave1 'bash -s' < first_slave_config.sh $NODES $DATASET &
    ssh slave2 'bash -s' < second_slave_config.sh $NODES $DATASET
fi

echo "The $NODES-node training has been completed!"