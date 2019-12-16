
#include <iostream>
#include <string>
#include "ShockTube.cuh"

#define fail " \033[1;31m"
#define pass " \033[1;32m"
#define reset "\033[0m"
#define cudaErrorCheck(call)                                \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("\n%s (%d): %s%s%s\n", __FILE__, __LINE__, fail, err_str, reset);\
  }                                                         \
}
// Wrap device CUDA calls with cucheck_err as in the following example.
// cudaErrorCheck(cudaGetLastError());


void ShockTube::DeviceTest01() {
	const std::string test = "DeviceMemoryAllocatedAndInitializedCorrectly";
	std::cout << __func__;
	nbrOfGrids = 10;
	allocDeviceMemory();
	initDeviceMemory<<<1,16>>>(nbrOfGrids, d_u1, d_u2, d_u3, d_vol, d_h, d_length,
		d_gama, d_cfl, d_nu, d_tau, d_step, d_cMax);
	cudaErrorCheck(cudaDeviceSynchronize());
	allocHostMemory();
	copyDeviceToHost(nbrOfGrids);
	freeDeviceMemory();
	double* Averages = getAverages();
	freeHostMemory();
	//std::cout << std::endl << Averages[0] << "  " << Averages[1] << "  " << Averages[2] << "  " << Averages[3] << std::endl; /**/
	double eps = 1e-14;
	if ((abs(Averages[0] - 0.65) < eps)
		&& (abs(Averages[1] - 0) < eps)
		&& (abs(Averages[2] - 1.6) < eps)
		&& (abs(Averages[3] - 0.64) < eps))
	std::cout << pass << test << reset << std::endl;
	else
	std::cout << fail << test << reset << std::endl;
}

