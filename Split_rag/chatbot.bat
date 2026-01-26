@echo off
REM Define virtual environment directory
set "VENV_DIR=.venv"

ollama --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Ollama not found. Installing silently...
    powershell -Command "Invoke-WebRequest -Uri https://ollama.com/download/OllamaSetup.exe -OutFile \"%~dp0OllamaSetup.exe\" -Verbose"

    start /wait "" OllamaSetup.exe /S
    del OllamaSetup.exe

    echo Ollama installed.
    echo PATH changes may require reopening this terminal.
    timeout /t 3 >nul
) ELSE (
    echo Ollama already installed:
    ollama --version
)

ollama list | findstr /i "granite4:1b-h" >nul
IF ERRORLEVEL 1 (
    echo Granite model not found. Pulling now...
    ollama pull granite4:1b-h
) ELSE (
    echo Granite model already installed.
)

IF NOT EXIST "%VENV_DIR%\Scripts\activate" (
    python -m venv %VENV_DIR%
)


CALL "%VENV_DIR%\Scripts\activate"

echo installing dependencies...
python -m pip install -r requirements.txt -q


python main.py

pause