cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_BUILD_TYPE release)
project(ncnn_lession11)

set(PY_WRAP OFF)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_subdirectory(src)
if(PY_WRAP)
    add_subdirectory(python)
else()
    add_subdirectory(examples)
endif()
