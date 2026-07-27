@echo off
setlocal

set BASE_PATH=

call .venv\Scripts\activate.bat

echo Running fp8 scale diagnostic...
echo. > fp8_diagnostic_output.txt
python check_fp8_scale.py "%BASE_PATH%" >> fp8_diagnostic_output.txt 2>&1

echo.
echo Done. Opening fp8_diagnostic_output.txt ...
notepad fp8_diagnostic_output.txt
