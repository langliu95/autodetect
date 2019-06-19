#!/bin/bash

# download stanfard-ner
echo "Downloading file stanford-ner-2018-10-16.zip"
# NOTE: need to update link for further versions
wget http://nlp.stanford.edu/software/stanford-ner-2018-10-16.zip
echo "Unpacking stanford-ner-2018-10-16.zip"
unzip stanford-ner-2018-10-16.zip

mkdir stanford-ner
cp stanford-ner-2018-10-16/stanford-ner.jar stanford-ner/stanford-ner.jar
cp stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz stanford-ner/english.all.3class.distsim.crf.ser.gz
cp stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.prop stanford-ner/english.all.3class.distsim.prop

echo "Clearing all"
rm -rf stanford-ner-2018-10-16 stanford-ner-2018-10-16.zip

# install dependencies
pip install contractions inflect pysrt
conda install nltk=3.4.1

python preprocessing.py