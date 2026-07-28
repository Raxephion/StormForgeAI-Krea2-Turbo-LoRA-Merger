@echo off
setlocal

set BASE_PATH=SET MODEL PATH HERE

call .venv\Scripts\activate.bat

echo Running metadata diagnostic...
echo. > metadata_diagnostic_output.txt
python check_metadata.py "%BASE_PATH%" >> metadata_diagnostic_output.txt 2>&1

echo.
echo Done. Opening metadata_diagnostic_output.txt ...
notepad metadata_diagnostic_output.txt
