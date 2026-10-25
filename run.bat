@echo off
REM Windows: create venv, install requirements, run the full experiment
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
python project28_ai4i_autoencoder_mlp.py %*
pause
