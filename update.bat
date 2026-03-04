@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: TrainStation — Windows Updater
:: Pulls latest from GitHub, reinstalls deps, rebuilds frontend.
:: ============================================================

title TrainStation Updater

:: Generate real ESC character for ANSI colors (Windows 10+)
for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
set "GREEN=%ESC%[32m"
set "YELLOW=%ESC%[33m"
set "RED=%ESC%[31m"
set "CYAN=%ESC%[36m"
set "RESET=%ESC%[0m"

echo.
echo %CYAN%============================================%RESET%
echo %CYAN%  TrainStation — Updater%RESET%
echo %CYAN%============================================%RESET%
echo.

cd /d "%~dp0"

:: ============================================================
:: 1. Check prerequisites
:: ============================================================
echo %CYAN%[1/4]%RESET% Checking prerequisites...

where git >nul 2>&1
if errorlevel 1 (
    echo %RED%ERROR: git not found. Install Git from https://git-scm.com/%RESET%
    goto :error_exit
)

if not exist "venv\Scripts\activate.bat" (
    echo %RED%ERROR: Virtual environment not found. Run install.bat first.%RESET%
    goto :error_exit
)

echo   %GREEN%OK%RESET%
echo.

:: ============================================================
:: 2. Pull latest changes
:: ============================================================
echo %CYAN%[2/4]%RESET% Pulling latest changes from GitHub...

:: Check for uncommitted changes
git diff --quiet 2>nul
if errorlevel 1 (
    echo   %YELLOW%WARNING: You have local changes. Stashing them...%RESET%
    git stash
    set "STASHED=1"
) else (
    set "STASHED=0"
)

git pull --ff-only
if errorlevel 1 (
    echo   %YELLOW%Fast-forward failed. Trying rebase...%RESET%
    git pull --rebase
    if errorlevel 1 (
        echo %RED%ERROR: Could not pull latest changes. You may have conflicting local modifications.%RESET%
        if "!STASHED!"=="1" (
            echo   Restoring your stashed changes...
            git stash pop
        )
        goto :error_exit
    )
)

if "!STASHED!"=="1" (
    echo   Restoring your local changes...
    git stash pop
    if errorlevel 1 (
        echo   %YELLOW%WARNING: Could not auto-restore stashed changes. Run 'git stash pop' manually.%RESET%
    )
)

echo   %GREEN%Done.%RESET%
echo.

:: ============================================================
:: 3. Update Python dependencies
:: ============================================================
echo %CYAN%[3/4]%RESET% Updating Python dependencies...

call venv\Scripts\activate.bat
pip install -r requirements\base.txt --quiet
if errorlevel 1 (
    echo %RED%ERROR: Failed to update dependencies.%RESET%
    goto :error_exit
)
echo   %GREEN%Done.%RESET%
echo.

:: ============================================================
:: 4. Rebuild frontend
:: ============================================================
echo %CYAN%[4/4]%RESET% Rebuilding frontend...

where npm >nul 2>&1
if errorlevel 1 (
    echo   %YELLOW%WARNING: npm not found — skipping frontend build.%RESET%
    echo   Install Node.js from https://nodejs.org/ then run:
    echo     cd ui\frontend ^&^& npm install ^&^& npm run build
    goto :update_done
)

cd ui\frontend
call npm install --silent 2>nul
call npm run build
if errorlevel 1 (
    echo   %YELLOW%WARNING: Frontend build failed. Try: cd ui\frontend ^&^& npm install ^&^& npm run build%RESET%
) else (
    echo   %GREEN%Frontend built.%RESET%
)
cd ..\..

:update_done
echo.
echo %GREEN%============================================%RESET%
echo %GREEN%  Update complete!%RESET%
echo %GREEN%============================================%RESET%
echo.
echo   Start the UI with: %CYAN%start_ui.bat%RESET%
echo.
goto :end

:error_exit
echo.
echo %RED%Update failed. See errors above.%RESET%
echo.

:end
pause
endlocal
