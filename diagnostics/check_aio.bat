@echo off
setlocal

set MODEL_PATH=SET MODEL PATH HERE

call .venv\Scripts\activate.bat

echo Running diagnostics on the all-in-one checkpoint...
echo. > aio_diagnostic_output.txt

echo ===== KEY SKELETON ===== >> aio_diagnostic_output.txt
python check_keys_full.py "%MODEL_PATH%" >> aio_diagnostic_output.txt 2>&1

echo. >> aio_diagnostic_output.txt
echo ===== METADATA ===== >> aio_diagnostic_output.txt
python check_metadata.py "%MODEL_PATH%" >> aio_diagnostic_output.txt 2>&1

echo.
echo Done. Opening aio_diagnostic_output.txt ...
notepad aio_diagnostic_output.txt
