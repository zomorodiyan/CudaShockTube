
#include <iostream>
#include <string>
#include "ShockTube.cuh"

#define fail " \033[1;31m"
#define pass " \033[1;32m"
#define yellow " \033[1;33m"
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
	const std::string test = "Memory Allocation And Initialization";
	std::cout << yellow << __func__ << reset;
	nbrOfGrids = 10;
	allocDeviceMemory();
	initDeviceMemory<<<1,16>>>(nbrOfGrids, d_u1, d_u2, d_u3, d_vol, d_h, d_length,
		d_gama, d_cfl, d_nu, d_tau, d_cMax, d_t);
	cudaErrorCheck(cudaDeviceSynchronize());
	allocHostMemory();
	copyDeviceToHost(nbrOfGrids);
	freeDeviceMemory();
	updateAverages();
	freeHostMemory();
	//std::cout << std::endl << Averages[0] << "  " << Averages[1] << "  " << Averages[2] << "  " << Averages[3] << std::endl; /**/
	double eps = 1e-14;
	if ((abs(roAverage - 0.5625) < eps)
		&& (abs(uAverage - 0) < eps)
		&& (abs(eAverage - 1.375) < eps)
		&& (abs(pAverage - 0.55) < eps))
	std::cout << pass << test << reset << std::endl;
	else
	std::cout << fail << test << reset << std::endl;
}

void ShockTube::DeviceTest02() {
	const std::string test = "Boundary Condition";
	std::cout << yellow << __func__ << reset;
	nbrOfGrids = 10;
	allocHostMemory();
	initHostMemory();
	u1[1] = u1[8] = 1; 	u2[1] = u2[8] = -1;	u3[1] = u3[8] = 1; 
	allocDeviceMemory();
	copyHostToDevice(nbrOfGrids);
	DeviceBoundaryCondition<<<1,16>>>(nbrOfGrids, d_u1, d_u2, d_u3);
	copyDeviceToHost(nbrOfGrids);
	if((1 == u1[0]) && (1 == u1[9]) && (1 == u2[0])
		&& (1 == u2[9]) && (1 == u3[0]) && (1 == u3[9]))
		std::cout << pass << test << reset << std::endl;
	else
		std::cout << fail << test << reset << std::endl;
}

