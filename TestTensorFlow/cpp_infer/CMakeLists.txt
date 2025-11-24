cmake_minimum_required(VERSION 3.16)
project(tf_face_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorFlow C API
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_executable(tf_face_infer
        facedetectcnn-data.cpp
        facedetectcnn-model.cpp
        facedetectcnn.cpp
        facedetectcnn.h
        main.cpp
        facedetection_export.h)
target_link_libraries(tf_face_infer ${OpenCV_LIBS} tensorflow)
