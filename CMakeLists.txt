cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(dev_cudaMalloc_test LANGUAGES CXX CUDA)

# Set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA and C++ compiler flags
set(CXX_FLAGS -Wall -m64 -O3)

set(CMAKE_CXX_FLAGS "-O3 -Wall -m64 -fPIC")

set(CMAKE_CUDA_FLAGS "-arch=sm_80 -O3 -rdc=true -lcudadevrt -lcuda")

add_executable(dev_cudaMalloc_test main.cu)
target_link_libraries(dev_cudaMalloc_test cuda)
set_property(TARGET dev_cudaMalloc_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

