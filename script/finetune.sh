#!/bin/bash

cd /home/workspace/freq_adapter

python finetune.py --lr 1e-4 --peft_type freq_adapter --enable_mcfa --enable_mgfa --use_freq_adapter

python finetune.py --lr 1e-4 --peft_type freq_adapter --enable_mcfa --enable_mgfa

python finetune.py --lr 1e-4 --peft_type freq_adapter --enable_mcfa --use_freq_adapter

python finetune.py --lr 1e-4 --peft_type freq_adapter  --enable_mgfa --use_freq_adapter