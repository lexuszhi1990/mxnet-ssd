cmake_minimum_required(VERSION 2.8.12)

set(MXNET_INCLUDE_DIRS "/mxnet/include")
set(MXNET_LIBRARY_ROOT "/mxnet/lib")
project(ssd-pedestrian C CXX)
add_definitions("-std=c++11")

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videostab)

include_directories(${MXNET_INCLUDE_DIRS})
link_directories(${MXNET_LIBRARY_ROOT})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ssd-pedestrian main.cc)
target_link_libraries(ssd-pedestrian ${OpenCV_LIBS} )
target_link_libraries(ssd-pedestrian mxnet)
