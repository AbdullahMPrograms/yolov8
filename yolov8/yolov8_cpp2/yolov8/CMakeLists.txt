# yolov8/CMakeLists.txt

# Define the executable target from the source files in this directory
add_executable(yolov8_demo
    main.cpp
    yolov8_utils.cpp
)

# Link the executable against all the required libraries
target_link_libraries(yolov8_demo
    PRIVATE
    ${ORT_LIBRARY}
    ${OpenCV_LIBS}      # <-- This now uses our manually defined variable from the parent
    ${TORCH_LIBRARIES}
    glog::glog
)

# Set include directories for this specific target
target_include_directories(yolov8_demo
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS} # <-- This now uses our manually defined variable from the parent
    ${TORCH_INCLUDE_DIRS}
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