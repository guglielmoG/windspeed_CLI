#!/bin/bash
pip install libs/retinanet/ --user
apt-get install -y libgl1-mesa-glx
pip install --no-cache-dir -r requirements.txt

mkdir models
mkdir out

gdown 'https://drive.google.com/uc?id=1RmbjBvG63SNEOlvCyCy9472fCZ023YAl' -O models/retinanet_model.h5

gdown 'https://drive.google.com/uc?id=1f24auOhBnCN7KYVTWG5zs2uxax8-S-Zs' -O models/efficientnet_final.h5

