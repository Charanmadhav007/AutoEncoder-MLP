#!/usr/bin/env bash
# Linux / macOS: create venv, install requirements, run the full experiment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python project28_ai4i_autoencoder_mlp.py "$@"
