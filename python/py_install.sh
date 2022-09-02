#!/bin/bash

python3.8 -m pip install --upgrade pip
python3.8 -m pip install pip-tools
python3.8 -m pip install -r python/requirements.txt
python3.8 -m pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html