#!/bin/bash

DATE=$(date +"%m%d_%H%M")

raspistill -vf -hf -k -o /home/pi/stoRoc/testFiles/$DATE.jpg
