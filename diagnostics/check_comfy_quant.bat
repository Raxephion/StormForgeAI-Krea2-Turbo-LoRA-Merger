@echo off
setlocal

set MODEL_PATH=SET MODEL PATH HERE

call .venv\Scripts\activate.bat

python check_comfy_quant.py "%MODEL_PATH%" > comfy_quant_output.txt 2>&1

echo Done. Opening comfy_quant_output.txt ...
notepad comfy_quant_output.txt
