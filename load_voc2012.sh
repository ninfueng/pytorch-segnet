#!/usr/bin/bash
DATA_LOC=./data

echo "Install Pascal VOC 2012."
mkdir $DATA_LOC
cd $DATA_LOC
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar -xvf ./VOCtrainval_11-May-2012.tar
rm ./VOCtrainval_11-May-2012.tar
cd -

