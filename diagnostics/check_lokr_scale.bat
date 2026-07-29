@echo off
setlocal

set LORA_PATH=SET MODEL PATH HERE

call .venv\Scripts\activate.bat

python check_lokr_scale.py "%LORA_PATH%" > lokr_scale_output.txt 2>&1

echo Done. Opening lokr_scale_output.txt ...
notepad lokr_scale_output.txt
