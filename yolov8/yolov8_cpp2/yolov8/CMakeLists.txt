set(GLOG_LIB_DIR "C:/dell/RyzenAI-SW/RyzenAI-SW/tutorial/yolov8/yolov8/yolov8_cpp/lib")

# Define the executable target from the source files in this directory
add_executable(yolov8_demo
    main.cpp
    yolov8_utils.cpp
)

# Link the executable against all the required libraries
target_link_libraries(yolov8_demo
    PRIVATE
    # ONNX Runtime and its dependencies
    ${ORT_LIBRARY}
    
    # OpenCV Libraries (from our manual definition)
    ${OpenCV_LIBS}
    
    # LibTorch Libraries
    ${TORCH_LIBRARIES}
    
    # Glog and Gflags (linked directly using the new variable)
    "${GLOG_LIB_DIR}/glog.lib"
    "${GLOG_LIB_DIR}/gflags.lib"

    # Windows libraries that glog/gflags might need
    Shlwapi.lib
    Dbghelp.lib
)

# Set include directories for this specific target
target_include_directories(yolov8_demo
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    "${GLOG_GFLAGS_ROOT}/include"  # <-- Use the new variable for the include path
)

# Set rpath for Linux/macOS to find shared libraries at runtime
if(UNIX AND NOT APPLE)
    set_target_properties(yolov8_demo PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib"
    )
endif()

# Add compiler options for this target
if(MSVC)
    target_compile_options(yolov8_demo PRIVATE /W4 /wd4267 /wd4251 /wd4244)
else()
    target_compile_options(yolov8_demo PRIVATE -Wno-unused-parameter)
endif()