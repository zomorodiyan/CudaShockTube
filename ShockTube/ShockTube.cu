#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <iostream>
#include "ShockTube.cuh"


#define cucheck_dev(call)                                   \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
    assert(0);                                              \
  }                                                         \
}
// Wrap device CUDA calls with cucheck_err as in the following example.
// cucheck_dev(cudaGetLastError());


// Allocate space for device copies of the variables
void ShockTube::allocDeviceMemory() {
	int size = nbrOfGrids * sizeof(double);
	cucheck_dev(cudaMalloc((void **)&d_u1, size));
	cucheck_dev(cudaMalloc((void **)&d_u2, size));
	cucheck_dev(cudaMalloc((void **)&d_u3, size));
	cucheck_dev(cudaMalloc((void **)&d_f1, size));
	cucheck_dev(cudaMalloc((void **)&d_f2, size));
	cucheck_dev(cudaMalloc((void **)&d_f3, size));
	cucheck_dev(cudaMalloc((void **)&d_vol, size));
}
