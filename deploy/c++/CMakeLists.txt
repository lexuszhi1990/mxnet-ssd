##******************************************************************************
##***************************** user definition ********************************
# set(CBLAS_INCLUDE_DIRS "/usr/local/Cellar/openblas/0.2.18_2/include")
# if use caffe as backend, you should set:
# set(CAFFE_INCLUDE_DIRS "/Users/tornadomeet/project/caffe/include")
# set(CAFFE_LIBRARY_ROOT "/Users/tornadomeet/project/caffe/build/lib")
# set(PROTOBUF_INCLUDE_DIRS "/usr/local/opt/protobuf@2.6/include")
# if use mxnet as backend
set(MXNET_INCLUDE_DIRS "/mxnet/include")
set(MXNET_LIBRARY_ROOT "/mxnet/lib")

##******************************************************************************
##**************************** cmake body **************************************
cmake_minimum_required(VERSION 2.8.12)
project(ssd-pedestrian C CXX)
add_definitions("-std=c++11")
# set(CMAKE_BUILD_TYPE Debug)  # if set debug model

include(cmake/Utils.cmake)
mtcnn_option(CPU_ONLY        "Build caffe with only cpu support" ON)

if(CPU_ONLY)
  add_definitions(-DCPU_ONLY)
endif()

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videostab)
if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

include_directories(${MXNET_INCLUDE_DIRS})
link_directories(${MXNET_LIBRARY_ROOT})

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ssd-pedestrian main.cc)
target_link_libraries(ssd-pedestrian ${OpenCV_LIBS} )
target_link_libraries(ssd-pedestrian mxnet)
