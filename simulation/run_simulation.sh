#!/bin/bash

# run simulations described in the paper

# install dependencies
conda install pomegranete  # 0.10.0
conda install statsmodels=0.9.0


NUM=20
for rep in {1..200}  # repetition
do
    echo "Repetition $rep."
    echo "Running experiments for linear regression models."
    python run_simulation.py linear 1000 500 100 $rep --num_change $NUM
    python run_simulation.py linear 1000 500 100 $rep --postfix "_p20" --num_change $NUM
    python run_simulation.py linear 1000 500 100 $rep --postfix "_idx" --num_change $NUM
    python run_simulation.py linear 1000 500 100 $rep --postfix "_idx_wrong" --num_change $NUM

    echo "Running experiments for ARMA models."
    python run_simulation.py arma 1000 500 3 $rep --dim_ma 2 --seed=11919 --num_change $NUM
    python run_simulation.py arma 5000 2500 6 $rep --dim_ma 5 --seed=11519 --num_change $NUM
    python run_simulation.py arma 1000 500 3 $rep --dim_ma 2 --seed=11919 --num_change $NUM --postfix "_idx" --num_change $NUM
    python run_simulation.py arma 5000 2500 6 $rep --dim_ma 5 --seed=11519 --postfix "_idx" --num_change $NUM

    echo "Running experiments for text topic models."
    python run_simulation.py brown 5000 2500 3 $rep --num_change $NUM

    echo "Running experiments for HMMs."
    python run_simulation.py hmm 5000 2500 3 $rep --num_change $NUM
    python run_simulation.py hmm 5000 2500 3 $rep --single --num_change $NUM

    echo "Running experiments for autograd-test-CuSum."
    python run_simulation.py autocusum 2000 1000 10 $rep --train_size 1000 --thresh 21.1638544850275 --num_change $NUM
    python run_simulation.py autocusum 10000 5000 100 $rep --train_size 5000 --thresh 127.3222457774553 --num_change $NUM
done