#!/bin/bash

cd /home/workspace/freq_adapter


python finetune.py --lr 1e-4 --peft_type freq_adapter --enable_mcfa --enable_mgfa --use_freq_adapter --model_name_or_path openai/clip-vit-large-patch14

python finetune.py --lr 1e-4 --peft_type freq_adapter  --enable_mcfa --enable_mgfa --use_freq_adapter --model_name_or_path openai/clip-vit-base-patch16