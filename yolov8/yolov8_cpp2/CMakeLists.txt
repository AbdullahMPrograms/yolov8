cmake_minimum_required(VERSION 3.16) # Increased version for better find_package support
project(
  yolov8_project # Renamed project to avoid conflict with subdirectory
  VERSION 1.0.0
  LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

# =================================================================
set(GLOG_GFLAGS_ROOT "C:/dell/RyzenAI-SW/RyzenAI-SW/tutorial/yolov8/yolov8/yolov8_cpp")
message(STATUS "Using custom glog/gflags from: ${GLOG_GFLAGS_ROOT}")
# =================================================================

# --- MSVC Compiler Flags (from your file) ---  
if(MSVC)
  string(APPEND CMAKE_CXX_FLAGS " /Zc:__cplusplus")
  # unreferenced formal parameter
  string(APPEND CMAKE_CXX_FLAGS " /wd4100")
  # std::codecvt deprecated warning
  string(APPEND CMAKE_CXX_FLAGS " /wd4996")
  # Disable "treat warnings as errors"
  string(APPEND CMAKE_CXX_FLAGS " /WX-")
  set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
endif()


# --- Ryzen AI / ONNX Runtime Setup (from your file) ---
set (RYZEN_AI_INSTALLATION_PATH $ENV{RYZEN_AI_INSTALLATION_PATH})

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "${RYZEN_AI_INSTALLATION_PATH}/onnxruntime")
  else()
    set(ONNXRUNTIME_ROOTDIR "/usr/local")
  endif()
endif()

# Global include and link directories for ONNX Runtime
include_directories("${ONNXRUNTIME_ROOTDIR}/include"
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

# Find the ONNX Runtime library file and store its path in ORT_LIBRARY 
find_library(ORT_LIBRARY onnxruntime HINTS "${ONNXRUNTIME_ROOTDIR}/lib" REQUIRED)
message(STATUS "Found ONNX Runtime Library: ${ORT_LIBRARY}")


# --- Dependency Finding ---

# 1. Find glog/gflags
#find_package(glog REQUIRED)

# 2. Define OpenCV paths manually (NO find_package)
# --- BEGIN OPENCV MANUAL DEFINITION ---
set(OpenCV_INCLUDE_DIRS "C:/opencv/mybuild/build/install/include")
set(OpenCV_LIBS
    "C:/opencv/mybuild/build/install/lib/opencv_core470.lib"
    "C:/opencv/mybuild/build/install/lib/opencv_highgui470.lib"
    "C:/opencv/mybuild/build/install/lib/opencv_imgproc470.lib"
    "C:/opencv/mybuild/build/install/lib/opencv_videoio470.lib"

)
message(STATUS "OpenCV paths are set manually. Bypassing find_package.")
# --- END OPENCV MANUAL DEFINITION ---

# 3. Find LibTorch
find_package(Torch REQUIRED)
message(STATUS "Found Torch libraries: ${TORCH_LIBRARIES}")

# This command tells CMake to look inside the 'yolov8' directory
add_subdirectory(yolov8)