@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: TrainStation -Windows Installer
:: Creates a venv, installs PyTorch with CUDA, then all deps.
:: ============================================================

title TrainStation Installer

:: Generate real ESC character for ANSI colors (Windows 10+)
for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
set "GREEN=%ESC%[32m"
set "YELLOW=%ESC%[33m"
set "RED=%ESC%[31m"
set "CYAN=%ESC%[36m"
set "RESET=%ESC%[0m"

echo.
echo %CYAN%============================================%RESET%
echo %CYAN%  TrainStation -Dependency Installer%RESET%
echo %CYAN%============================================%RESET%
echo.

cd /d "%~dp0"

:: ============================================================
:: 1. Find Python
:: ============================================================
echo %CYAN%[1/6]%RESET% Locating Python...

set "PYTHON="

:: Try python in PATH first
where python >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do set "PY_VER=%%i"
    echo   Found: !PY_VER!
    set "PYTHON=python"
    goto :check_python_version
)

:: Try py launcher
where py >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%i in ('py -3 --version 2^>^&1') do set "PY_VER=%%i"
    echo   Found: !PY_VER!
    set "PYTHON=py -3"
    goto :check_python_version
)

echo %RED%ERROR: Python not found.%RESET%
echo   Install Python 3.10+ from https://www.python.org/downloads/
echo   Make sure to check "Add Python to PATH" during installation.
goto :error_exit

:check_python_version
%PYTHON% -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
if errorlevel 1 (
    echo %RED%ERROR: Python 3.10 or newer is required. Found: !PY_VER!%RESET%
    goto :error_exit
)
echo   %GREEN%OK%RESET% -!PY_VER!
echo.

:: ============================================================
:: 2. Create or reuse venv
:: ============================================================
echo %CYAN%[2/6]%RESET% Setting up virtual environment...

if exist "venv\Scripts\activate.bat" (
    echo   Existing venv found -reusing it.
    goto :activate_venv
)

echo   Creating new venv...
%PYTHON% -m venv venv
if errorlevel 1 (
    echo %RED%ERROR: Failed to create virtual environment.%RESET%
    goto :error_exit
)
echo   %GREEN%Created.%RESET%

:activate_venv
call venv\Scripts\activate.bat
echo   %GREEN%Activated.%RESET%
echo.

:: ============================================================
:: 3. Upgrade pip
:: ============================================================
echo %CYAN%[3/6]%RESET% Upgrading pip...
python -m pip install --upgrade pip --quiet
echo   %GREEN%Done.%RESET%
echo.

:: ============================================================
:: 4. Choose CUDA or CPU
:: ============================================================
echo %CYAN%[4/6]%RESET% PyTorch installation
echo.
echo   Which PyTorch variant do you want to install?
echo.
echo     1) CUDA 12.8  (NVIDIA GPU -recommended)
echo     2) CUDA 12.4  (NVIDIA GPU -older drivers)
echo     3) CPU only   (no GPU acceleration)
echo     4) Skip       (PyTorch already installed)
echo.
set /p "TORCH_CHOICE=  Enter choice [1-4] (default: 1): "
if "!TORCH_CHOICE!"=="" set "TORCH_CHOICE=1"

if "!TORCH_CHOICE!"=="1" goto :torch_cu128
if "!TORCH_CHOICE!"=="2" goto :torch_cu124
if "!TORCH_CHOICE!"=="3" goto :torch_cpu
if "!TORCH_CHOICE!"=="4" goto :torch_skip
echo %YELLOW%Invalid choice. Defaulting to CUDA 12.8.%RESET%
goto :torch_cu128

:torch_cu128
echo.
echo   Installing PyTorch with CUDA 12.8...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :torch_error
goto :torch_done

:torch_cu124
echo.
echo   Installing PyTorch with CUDA 12.4...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto :torch_error
goto :torch_done

:torch_cpu
echo.
echo   Installing PyTorch (CPU only)...
pip install torch torchvision
if errorlevel 1 goto :torch_error
goto :torch_done

:torch_skip
echo.
echo   Skipping PyTorch installation.
python -c "import torch; print(f'  Found torch {torch.__version__}')" 2>nul
if errorlevel 1 echo   %YELLOW%WARNING: torch not found -you may need to install it manually.%RESET%
goto :torch_done

:torch_error
echo %RED%ERROR: PyTorch installation failed.%RESET%
goto :error_exit

:torch_done
echo   %GREEN%Done.%RESET%
echo.

:: ============================================================
:: 5. Install base requirements
:: ============================================================
echo %CYAN%[5/6]%RESET% Installing dependencies...
pip install -r requirements\base.txt
if errorlevel 1 (
    echo %RED%ERROR: Failed to install base requirements.%RESET%
    goto :error_exit
)
echo   %GREEN%Base dependencies installed.%RESET%

:: Optional optimizers
echo.
echo   Install optional optimizers (bitsandbytes, prodigy, lion, came, schedulefree)?
set /p "OPT_CHOICE=  [y/N]: "
if /i "!OPT_CHOICE!"=="y" (
    echo   Installing optional optimizers...
    pip install -r requirements\optimizers.txt
    echo   %GREEN%Optional optimizers installed.%RESET%
)
echo.

:: ============================================================
:: 6. Build frontend
:: ============================================================
echo %CYAN%[6/6]%RESET% Building frontend...

where npm >nul 2>&1
if errorlevel 1 (
    echo   %YELLOW%WARNING: npm not found -skipping frontend build.%RESET%
    echo   Install Node.js from https://nodejs.org/ then run:
    echo     cd ui\frontend ^&^& npm install ^&^& npm run build
    goto :frontend_done
)

cd ui\frontend
call npm install --silent 2>nul
call npm run build
if errorlevel 1 (
    echo   %YELLOW%WARNING: Frontend build failed. The server will still work%RESET%
    echo   %YELLOW%but the UI won't load. Try: cd ui\frontend ^&^& npm install ^&^& npm run build%RESET%
) else (
    echo   %GREEN%Frontend built.%RESET%
)
cd ..\..

:frontend_done
echo.

:: ============================================================
:: Done
:: ============================================================
echo %GREEN%============================================%RESET%
echo %GREEN%  Installation complete!%RESET%
echo %GREEN%============================================%RESET%
echo.
echo   To start the UI:
echo     %CYAN%start_ui.bat%RESET%
echo.
echo   Or manually:
echo     %CYAN%venv\Scripts\activate.bat%RESET%
echo     %CYAN%python run_ui.py%RESET%
echo.
echo   To train from CLI:
echo     %CYAN%venv\Scripts\activate.bat%RESET%
echo     %CYAN%python run.py --config your_config.yaml%RESET%
echo.
goto :end

:error_exit
echo.
echo %RED%Installation failed. See errors above.%RESET%
echo.

:end
pause
endlocal
