@echo off
echo RyzenAI 1.5 GA YOLOv8 Camera Demo
echo ===============================

rem RyzenAI 1.5 GA paths
set RYZEN_AI_PATH=C:\Program Files\RyzenAI\1.5.0
set PATH=%RYZEN_AI_PATH%\onnxruntime\bin;%cd%\..;C:\opencv\mybuild\build\install\bin;%PATH%;
set XLNX_VART_FIRMWARE=%RYZEN_AI_PATH%\voe-4.0-win_amd64\xclbins\strix\AMD_AIE2P_Nx4_Overlay.xclbin

echo Checking RyzenAI 1.5 GA installation...
if exist "%RYZEN_AI_PATH%\onnxruntime\bin\onnxruntime.dll" (
    echo ✓ ONNX Runtime found: %RYZEN_AI_PATH%\onnxruntime\bin\
) else (
    echo ✗ ERROR: ONNX Runtime not found at %RYZEN_AI_PATH%\onnxruntime\bin\
    echo Please verify RyzenAI 1.5 GA installation
    pause
    exit /b 1
)

if exist "%XLNX_VART_FIRMWARE%" (
    echo ✓ xclbin found: %XLNX_VART_FIRMWARE%
) else (
    echo ✗ ERROR: xclbin not found: %XLNX_VART_FIRMWARE%
    pause
    exit /b 1
)

echo.
echo Environment Variables:
set XLNX_ENABLE_TRACE=1
set DEBUG_DEMO=1
set XLNX_ONNX_EP_VERBOSE=1
echo XLNX_ENABLE_TRACE=1
echo DEBUG_DEMO=1  
echo XLNX_ONNX_EP_VERBOSE=1

echo.
echo Starting YOLOv8 with parameters: -c 1 -s 0 -x 1 -y 1 -D -R 1280x720 -r 1280x720
echo ================================================================================

%cd%\..\bin\camera_yolov8_nx1x4.exe -c 1 -s 0 -x 1 -y 1 -D -R 1280x720 -r 1280x720 %1

echo.
echo Application exited with code: %ERRORLEVEL%
if %ERRORLEVEL% neq 0 echo Check the output above for any error messages.
pause