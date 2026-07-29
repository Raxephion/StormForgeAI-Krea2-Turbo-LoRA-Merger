@echo off
setlocal

set LORA_PATH=SET MODEL PATH HERE
set BASE_PATH=SET MODEL PATH HERE


call .venv\Scripts\activate.bat

python check_lokr_shapes.py "%LORA_PATH%" "%BASE_PATH%" > lokr_shapes_output.txt 2>&1

echo Done. Opening lokr_shapes_output.txt ...
notepad lokr_shapes_output.txt
