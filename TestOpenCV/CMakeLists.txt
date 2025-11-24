# Minimum required CMake version
cmake_minimum_required(VERSION 3.28)
# Project name
project(TestTensorflow)

# Require C++17 or higher
set(CMAKE_CXX_STANDARD 17)
# Fail instead of downgrading if C++17+ is unavailable
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TensorFlow paths (WSL)
set(TENSORFLOW_INCLUDE_DIR /usr/local/include)
set(TENSORFLOW_LIBRARY /usr/local/lib/libtensorflow.so)

# Locate the system OpenCV libraries
find_package(OpenCV REQUIRED)

 # Add current directory to the header search path
include_directories(.)

# Define the executable target and its sources
add_executable(TestTensorflow
        facedetectcnn-data.cpp
        facedetectcnn-model.cpp
        facedetectcnn.cpp
        facedetectcnn.h
        main.cpp
        facedetection_export.h)

# Add TensorFlow and OpenCV include directories to the target
target_include_directories(TestTensorflow PRIVATE
        ${TENSORFLOW_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS})

# Link the TensorFlow and OpenCV libraries
target_link_libraries(TestTensorflow PRIVATE
        ${TENSORFLOW_LIBRARY}
        ${OpenCV_LIBS})