#!/bin/bash

NUM_USERS=100
SHARDS=2
SEED=$1 # already done: 0,1,2,3,4
while getopts s:n: option; do
  case "${option}" in
    s) SHARDS=${OPTARG};;
    n) NUM_USERS=${OPTARG};;
  esac
done

# "Best" case
python3 main_fed.py --seed $SEED --dataset coba --model cnn --num_classes 14 --log_level info --epochs 1000 --lr 0.1 --num_users $NUM_USERS --shard_per_user $SHARDS --frac 0.3 --local_ep 1 --local_bs 10 --results_save coba_fedavg_bestcase_run1

# "Average" case
python3 main_fed.py --seed $SEED --dataset coba --model cnn --num_classes 14 --log_level info --epochs 1000 --lr 0.1 --num_users $NUM_USERS --shard_per_user $SHARDS --frac 0.5 --local_ep 1 --local_bs 10 --results_save coba_fedavg_averagecase_run1

# "Worst" case
python3 main_fed.py --seed $SEED --dataset coba --model cnn --num_classes 14 --log_level info --epochs 1000 --lr 0.1 --num_users $NUM_USERS --shard_per_user $SHARDS --frac 1 --local_ep 1 --local_bs 10 --results_save coba_fedavg_worstcase_run1