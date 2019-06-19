#!/bin/bash

# run real data experiments described in the paper

# install dependencies
conda install pandas

# for experiments without shuffle
echo "Pair-wise experiments."
for pair in {0..63}
do
    python tv_shows.py $pair
done

# for experiments with shuffle
# download the dataset first
wget https://www.stat.washington.edu/~liu16/autodetect/tv_subtitles_autogradtest.zip
unzip tv_subtitles_autogradtest.zip

# running experiments
echo "Experiments with shuffle"
for rep in {1..200}
do
    echo "Repetition $rep."
    python tv_shows.py $rep --shuffle
done
