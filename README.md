# cudaMallocRuntime

### To build project

``` 
  git clone https://github.com/agalup/cudaMallocRuntime.git
  cd cudaMallocRuntime
  mkdir build
  cd build
  cmake .. && make
  ```

### To launch Multi-Process Service:

1. Set up the MPS environment variables:

``` 
  export CUDA_VISIBLE_DEVICES=0
  export CUDA_MPS_PIPE_DIRECTORY=/home/<username>/mps/mps
  export CUDA_MPS_LOG_DIRECTORY=/home/<username>/mps/log
  ```

2. Start the MPS control daemon:

```
  nvidia-cuda-mps-control -d
  ```
  
### To run project

```
  ./dev_cudaMalloc_test
  ```
 
Error to observe: 

```
  err(811) unrecognized error code
  ```

### To quit MPS:

```
  echo quit | nvidia-cuda-mps-control
  ```
  
  
