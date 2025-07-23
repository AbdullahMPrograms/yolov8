set RYZEN_AI_PATH=C:\Program Files\RyzenAI\1.5.0
set PATH=%RYZEN_AI_PATH%\onnxruntime\bin;%cd%\..;C:\opencv\mybuild\build\install\bin;%PATH%;
set XLNX_VART_FIRMWARE=%RYZEN_AI_PATH%\voe-4.0-win_amd64\xclbins\strix\AMD_AIE2P_Nx4_Overlay.xclbin

set XLNX_ENABLE_TRACE=1
set DEBUG_DEMO=1
set XLNX_ONNX_EP_VERBOSE=1

%cd%\..\bin\camera_yolov8_nx1x4.exe -c 1 -s 0 -x 1 -y 1 -D -R 1280x720 -r 2560x1440 %1