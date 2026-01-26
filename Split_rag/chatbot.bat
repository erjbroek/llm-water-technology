@echo off
REM Define virtual environment directory
set "VENV_DIR=.venv"

@REM ollama --version >nul 2>&1
@REM IF ERRORLEVEL 1 (
@REM     echo Ollama not found. Installing silently...
@REM     powershell -Command "Invoke-WebRequest -Uri https://ollama.com/download/OllamaSetup.exe -OutFile \"%~dp0OllamaSetup.exe\" -Verbose"

@REM     start /wait "" OllamaSetup.exe /S
@REM     del OllamaSetup.exe

@REM     echo Ollama installed.
@REM     echo PATH changes may require reopening this terminal.
@REM     timeout /t 3 >nul
@REM ) ELSE (
@REM     echo Ollama already installed:
@REM     ollama --version
@REM )

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