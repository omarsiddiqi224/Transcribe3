#!/bin/bash

python3 -m venv env
source env/bin/activate

pip install -r requirements.txt 
pipx install insanely-fast-whisper
pipx runpip insanely-fast-whisper install flash-attn --no-build-isolation
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade transformers optimum accelerate
