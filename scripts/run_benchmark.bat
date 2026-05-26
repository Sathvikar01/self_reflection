@echo off
REM ============================================================================
REM Benchmark Runner - Convenience Wrapper for Windows
REM ============================================================================
REM
REM Usage:
REM   run_benchmark.bat                    Run with defaults
REM   run_benchmark.bat --mock             Run in mock mode (no API)
REM   run_benchmark.bat --mock --num-questions 5   Quick test
REM   run_benchmark.bat --resume           Resume from checkpoint
REM
REM ============================================================================

setlocal

REM Get script directory
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Run the benchmark
python scripts\run_benchmark.py %*

endlocal
