set PATH=%ONNXRUNTIME_ROOTDIR%/bin;%cd%\..;C:\opencv\mybuild\build\install\bin;%PATH%;
set XLNX_VART_FIRMWARE=C:\Program Files\RyzenAI\1.5.0\voe-4.0-win_amd64\xclbins\strix\AMD_AIE2P_Nx4_Overlay.xclbin

set XLNX_ENABLE_TRACE=1
::set DEEPHI_PROFILING=1
set DEBUG_DEMO=1
%cd%\..\bin\camera_yolov8_nx1x4.exe -c 1 -s 0 -x 1 -y 1 -D -R 1280x720 -r 2560x1440 %1