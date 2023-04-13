#!/bin/bash

CONFIG_FILE="${config.yaml}"

export CUDA_VISIBLE_DEVICES=${DEVICE}

python main.py --config_file ${CONFIG_FILE} 