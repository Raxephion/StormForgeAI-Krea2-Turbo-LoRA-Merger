@echo off
setlocal

set BASE_PATH=C:\ComfyFAST\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\krea2_turbo_base_fp8_scaled.safetensors

call .venv\Scripts\activate.bat

echo Running metadata diagnostic...
echo. > metadata_diagnostic_output.txt
python check_metadata.py "%BASE_PATH%" >> metadata_diagnostic_output.txt 2>&1

echo.
echo Done. Opening metadata_diagnostic_output.txt ...
notepad metadata_diagnostic_output.txt
