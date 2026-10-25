@echo off
set PYTHON=python

if "%1"=="data" (
    %PYTHON% -m fifa_value.cli data
) else if "%1"=="features" (
    %PYTHON% -m fifa_value.cli features
) else if "%1"=="train" (
    %PYTHON% -m fifa_value.cli train
) else if "%1"=="evaluate" (
    %PYTHON% -m fifa_value.cli evaluate
) else if "%1"=="predict" (
    %PYTHON% -m fifa_value.cli predict
) else if "%1"=="test" (
    pytest tests/
) else if "%1"=="lint" (
    ruff check src/ tests/
) else if "%1"=="all" (
    %PYTHON% -m fifa_value.cli data
    %PYTHON% -m fifa_value.cli features
    %PYTHON% -m fifa_value.cli train
    %PYTHON% -m fifa_value.cli evaluate
    %PYTHON% -m fifa_value.cli predict
    pytest tests/
) else (
    echo Usage: make.bat [data^|features^|train^|evaluate^|predict^|test^|lint^|all]
)
