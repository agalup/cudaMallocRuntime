cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(dev_cudaMalloc_test LANGUAGES CXX CUDA)

# Set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

#find_package(CUDA QUIET REQUIRED)
find_package(CUDAToolkit QUIET REQUIRED)

# CUDA and C++ compiler flags
set(CXX_FLAGS -Wall -m64 -O3 -std=c++17)

set(CMAKE_CXX_FLAGS "-O3 -Wall -m64 -fPIC -std=c++17")
#set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

set(CMAKE_CUDA_FLAGS "-arch=sm_75 -Xcompiler -Wall -O3 -Xcudafe=--display_error_number -lineinfo --expt-extended-lambda -use_fast_math -Wno-deprecated-gpu-targets --expt-relaxed-constexpr -Xptxas -warn-spills -res-usage -rdc=true --ptxas-options=-v")

#set(CUDA_INCLUDE_DIRS /usr/local/cuda/include/ ${CUDA_INCLUDE_DIRS})
#message("CUDA_INCLUDE_DIRS " ${CUDA_INCLUDE_DIRS})
#message("cuda toolki include dirs" ${CUDAToolkit_INCLUDE_DIRS})

add_executable(dev_cudaMalloc_test main.cu)
#target_include_directories(dev_cudaMalloc_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
#target_compile_options(dev_cudaMalloc_test INTERFACE
#    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
#    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)
#PUBLIC ${CUDA_FLAGS})
#target_compile_options(dev_cudaMalloc_test 
#target_link_libraries(dev_cudaMalloc_test CUDA::cuda_driver CUDA::cudart)
set_property(TARGET dev_cudaMalloc_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

