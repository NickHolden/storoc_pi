#!/bin/bash

DATE=$(date +"%m%d_%H%M")

raspivid -vf -hf -k -o /home/pi/stoRoc/testFiles/$DATE.h264

MP4Box -add $DATE.h264 $DATE.mp4

rm $DATE.h264
