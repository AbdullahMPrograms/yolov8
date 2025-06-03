set ONNXRUNTIME_ROOTDIR=C:\Program Files\RyzenAI\1.4.1
set PATH=%ONNXRUNTIME_ROOTDIR%/bin;%cd%\..;C:\Users\abdul\OneDrive\Desktop\opencv\mybuild\build\lib\Release;C:\Users\abdul\OneDrive\Desktop\opencv\mybuild\build\bin\Release;%PATH%
set XLNX_VART_FIRMWARE=C:\Program Files\RyzenAI\1.4.1\voe-4.0-win_amd64\xclbins\strix\AMD_AIE2P_Nx4_Overlay.xclbin

%cd%\..\bin\test_jpeg_yolov8.exe ./DetectionModel_int.onnx ./sample_yolov8.jpg