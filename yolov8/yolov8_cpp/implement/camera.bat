@echo off
echo ========== RyzenAI YOLOv8 Camera Debug Session ==========
echo Timestamp: %date% %time%
echo Working Directory: %cd%

echo.
echo ========== Environment Setup ==========
set PATH=%ONNXRUNTIME_ROOTDIR%/bin;%cd%\..;C:\opencv\mybuild\build\install\bin;%PATH%;
echo PATH updated successfully

echo.
echo ========== RyzenAI Configuration ==========
set XLNX_VART_FIRMWARE=C:\Program Files\RyzenAI\1.5.0\voe-4.0-win_amd64\xclbins\strix\AMD_AIE2P_Nx4_Overlay.xclbin
echo XLNX_VART_FIRMWARE set to: %XLNX_VART_FIRMWARE%

echo Checking if xclbin file exists...
if exist "%XLNX_VART_FIRMWARE%" (
    echo ✓ xclbin file found: %XLNX_VART_FIRMWARE%
) else (
    echo ✗ ERROR: xclbin file NOT found: %XLNX_VART_FIRMWARE%
    echo This will likely cause initialization failure
    pause
)

echo.
echo ========== Debug Flags ==========
set XLNX_ENABLE_TRACE=1
echo XLNX_ENABLE_TRACE=1 (Vitis AI tracing enabled)

::set DEEPHI_PROFILING=1
set DEBUG_DEMO=1
echo DEBUG_DEMO=1 (Demo debug output enabled)

set GLOG_v=3
echo GLOG_v=3 (Verbose Google logging)

set GLOG_logtostderr=1
echo GLOG_logtostderr=1 (Log to stderr for immediate output)

set GLOG_minloglevel=0
echo GLOG_minloglevel=0 (Show all log levels)

set VITIS_AI_EP_DEBUG=1
echo VITIS_AI_EP_DEBUG=1 (Vitis AI execution provider debug)

echo.
echo ========== Executable Check ==========
set EXE_PATH=%cd%\..\bin\camera_yolov8_nx1x4.exe
echo Checking executable: %EXE_PATH%
if exist "%EXE_PATH%" (
    echo ✓ Executable found: %EXE_PATH%
) else (
    echo ✗ ERROR: Executable NOT found: %EXE_PATH%
    pause
    exit /b 1
)

echo.
echo ========== Model File Check ==========
set MODEL_PATH=%cd%\DetectionModel_int.onnx
echo Checking model file: %MODEL_PATH%
if exist "%MODEL_PATH%" (
    echo ✓ Model file found: %MODEL_PATH%
) else (
    echo ✗ WARNING: Model file NOT found: %MODEL_PATH%
)

echo.
echo ========== Starting YOLOv8 Application ==========
echo Command: %EXE_PATH% -c 2 -s 1 -x 1 -y 1 -D -R 1280x720 -r 1280x720 %1
echo Arguments explanation:
echo   -c 2: Number of parallel DPU threads ^(2 for inference^)
echo   -s 0: Input stream - camera device 0 ^(default camera^)
echo   -x 1: Intra-op num threads ^(1 thread within nodes^)
echo   -y 1: Inter-op num threads ^(1 thread across nodes^)
echo   -D: Disable thread spinning
echo   -R 1280x720: Camera capture resolution
echo   -r 1280x720: Display resolution
echo.
echo Press Ctrl+C to stop the application...
echo ==========================================

"%EXE_PATH%" -c 1 -s 1 -x 1 -y 1 -D -R 1280x720 -r 1280x720 %1

echo.
echo ========== Application Exited ==========
echo Exit code: %ERRORLEVEL%
if %ERRORLEVEL% equ 0 (
    echo ✓ Application completed successfully
) else (
    echo ✗ Application exited with error code: %ERRORLEVEL%
    echo Common error codes:
    echo   -1073741819 ^(0xC0000005^): Access violation ^(memory error^)
    echo   -1073740791 ^(0xC0000409^): Stack buffer overflow
    echo   -1073741515 ^(0xC0000135^): DLL not found
    echo   1: General error
)
echo.
pause

