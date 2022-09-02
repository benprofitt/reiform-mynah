#!/bin/bash

python3.8 -m pip install --upgrade pip
python3.8 -m pip install pip-tools
python3.8 -m pip install -r python/requirements.txt
python3.8 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu