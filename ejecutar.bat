@echo off
rem ============================================================================
rem Suite Educativa Metodos Computacionales - FaCENA UNNE
rem Script de 1 clic para Windows
rem ============================================================================
cd /d "%~dp0"

set "PY_EXE="

rem 1. Verificar entorno virtual local .venv o venv
if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
    goto :ENCONTRADO
)
if exist "venv\Scripts\python.exe" (
    set "PY_EXE=venv\Scripts\python.exe"
    goto :ENCONTRADO
)

rem 2. Verificar comando python en PATH
python --version >nul 2>&1
if not errorlevel 1 (
    set "PY_EXE=python"
    goto :ENCONTRADO
)

rem 3. Verificar launcher py
py --version >nul 2>&1
if not errorlevel 1 (
    set "PY_EXE=py"
    goto :ENCONTRADO
)

echo.
echo [ERROR] No se encontro Python en el sistema.
echo Instale Python 3.10+ y marque 'Add Python to PATH'.
echo.
pause
exit /b 1

:ENCONTRADO
echo ============================================================================
echo   Suite de Metodos Computacionales - FaCENA UNNE
echo   Python detectado: %PY_EXE%
echo ============================================================================
echo.
echo Verificando dependencias...
%PY_EXE% -c "import streamlit, sympy, numpy, pandas, plotly" >nul 2>&1
if errorlevel 1 (
    echo Instalando librerias faltantes desde requirements.txt...
    %PY_EXE% -m pip install -r requirements.txt
)

echo Iniciando aplicacion Streamlit...
%PY_EXE% -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo [ERROR] Ocurrio un problema al iniciar Streamlit.
    pause
    exit /b 1
)

pause