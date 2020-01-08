#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <iostream>
#include "ShockTube.cuh"


#define fail "\033[1;31m"
#define reset "\033[0m"
#define cudaErrorCheck(call)                                \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("\n%s (%d): %s%s%s\n", __FILE__, __LINE__, fail, err_str, reset);\
    assert(0);                                              \
  }                                                         \
}
// Wrap device CUDA calls with cucheck_err as in the following example.
// cudaErrorCheck(cudaGetLastError());


// Allocate space for device copies of the variables
void ShockTube::allocDeviceMemory() {
	int size = nbrOfGrids * sizeof(double);
	cudaErrorCheck(cudaMalloc((void **)&d_u1, size));
	cudaErrorCheck(cudaMalloc((void **)&d_u2, size));
	cudaErrorCheck(cudaMalloc((void **)&d_u3, size));
	cudaErrorCheck(cudaMalloc((void **)&d_u1Temp, size));
	cudaErrorCheck(cudaMalloc((void **)&d_u2Temp, size));
	cudaErrorCheck(cudaMalloc((void **)&d_u3Temp, size));
	cudaErrorCheck(cudaMalloc((void **)&d_f1, size));
	cudaErrorCheck(cudaMalloc((void **)&d_f2, size));
	cudaErrorCheck(cudaMalloc((void **)&d_f3, size));
	cudaErrorCheck(cudaMalloc((void **)&d_vol, size));
	cudaErrorCheck(cudaMalloc((void **)&d_h, sizeof(double)));
	cudaErrorCheck(cudaMalloc((void **)&d_length, sizeof(double)));
	cudaErrorCheck(cudaMalloc((void **)&d_gama, sizeof(double)));
	cudaErrorCheck(cudaMalloc((void **)&d_cfl, sizeof(double)));
	cudaErrorCheck(cudaMalloc((void **)&d_nu, sizeof(double)));
	cudaErrorCheck(cudaMalloc((void **)&d_tau, sizeof(double)));
	cudaErrorCheck(cudaMalloc((void **)&d_cMax, sizeof(double)));
	cudaErrorCheck(cudaMalloc((void **)&d_t, sizeof(double)));
}

// Free allocated space for device copies of the variables
void ShockTube::freeDeviceMemory() {
	cudaErrorCheck(cudaFree(d_u1));
	cudaErrorCheck(cudaFree(d_u2));
	cudaErrorCheck(cudaFree(d_u3));
	cudaErrorCheck(cudaFree(d_u1Temp));
	cudaErrorCheck(cudaFree(d_u2Temp));
	cudaErrorCheck(cudaFree(d_u3Temp));
	cudaErrorCheck(cudaFree(d_f1));
	cudaErrorCheck(cudaFree(d_f2));
	cudaErrorCheck(cudaFree(d_f3));
	cudaErrorCheck(cudaFree(d_vol));
	cudaErrorCheck(cudaFree(d_h));
	cudaErrorCheck(cudaFree(d_length));
	cudaErrorCheck(cudaFree(d_gama));
	cudaErrorCheck(cudaFree(d_cfl));
	cudaErrorCheck(cudaFree(d_nu));
	cudaErrorCheck(cudaFree(d_tau));
	cudaErrorCheck(cudaFree(d_cMax));
}

// calculate and update value of d_cMax
__device__ void updateCMax(const int nbrOfGrids, const double *d_u1,
	const double *d_u2, const double *d_u3, const double *d_gama,
	double *d_cMax) {
	*d_cMax = 0;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	double ro, p, u;
	__shared__ double c;
	for (int i = index; i < nbrOfGrids; i += stride){
		if (d_u1[i] == 0)
			continue;
		ro = d_u1[i];
		u = d_u2[i] / ro;
		p = (d_u3[i] - ro * u * u / 2) * (*d_gama - 1);
		c = sqrt(*d_gama * abs(p) / ro);
		if (*d_cMax < c + abs(u))
			*d_cMax = c + abs(u);
	}
}

// Assign Sod's shock tube problem initial conditions to device memory
__global__ void initDeviceMemory(const int nbrOfGrids, double *d_u1,
	double *d_u2, double *d_u3, double *d_vol, double *d_h,
	double *d_length, double *d_gama, double *d_cfl, double *d_nu,
	double *d_tau, double *d_cMax, double *d_t) {
	*d_t = 0;							// time
	*d_length = 1;					// length of shock tube
	*d_gama = 1.4;						// ratio of specific heats
	*d_cfl = 0.9;						// Courant-Friedrichs-Lewy number
	*d_nu = 0.0;							// artificial viscosity coefficient
	*d_h = *d_length / (nbrOfGrids - 1);  // space grid size
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x;
	for(int i = index; i < nbrOfGrids; i+= stride){
		double e, ro, p, u = 0;
		if (index < nbrOfGrids){
			if (index >= int(nbrOfGrids / 2)) { ro = 0.125, p = 0.1; }
			else { ro = 1, p = 1; }
			e = p / (*d_gama - 1) + ro * u * u / 2;
			d_u1[i] = ro;
			d_u2[i] = ro * u;
			d_u3[i] = e;
			d_vol[i] = 1;
		}
	}
	updateCMax(nbrOfGrids, d_u1, d_u2, d_u3, d_gama, d_cMax); 
	*d_tau = (*d_cfl) * (*d_h) / (*d_cMax);    // time grid size
}

// copy device data members to host data members
void ShockTube::copyDeviceToHost(const int nbrOfGrids) {
	int size = nbrOfGrids * sizeof(double);
	cudaErrorCheck(cudaMemcpy(u1, d_u1, size, cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(u2, d_u2, size, cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(u3, d_u3, size, cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(vol, d_vol, size, cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(&h, d_h, sizeof(double), cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(&length, d_length, sizeof(double), cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(&gama, d_gama, sizeof(double), cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(&cfl, d_cfl, sizeof(double), cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(&cMax, d_cMax, sizeof(double), cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(&nu, d_nu, sizeof(double), cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(&tau, d_tau, sizeof(double), cudaMemcpyDeviceToHost));
}

// copy flux from device to host (for debegging purpose)
void ShockTube::copyFluxFromDeviceToHost(const int nbrOfGrids) {
	int size = nbrOfGrids * sizeof(double);
	cudaErrorCheck(cudaMemcpy(f1, d_f1, size, cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(f2, d_f2, size, cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(f3, d_f3, size, cudaMemcpyDeviceToHost));
}


// copy host data members to device data members
void ShockTube::copyHostToDevice(const int nbrOfGrids) {
	int size = nbrOfGrids * sizeof(double);
	cudaErrorCheck(cudaMemcpy(d_u1, u1, size, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(d_u2, u2, size, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(d_u3, u3, size, cudaMemcpyHostToDevice));
	/*/
	cudaErrorCheck(cudaMemcpy(d_f1, f1, size, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(d_f2, f2, size, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(d_f3, f3, size, cudaMemcpyHostToDevice));
	/**/
	cudaErrorCheck(cudaMemcpy(d_vol, vol, size, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(d_h, &h, sizeof(double), cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(d_length, &length, sizeof(double), cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(d_gama, &gama, sizeof(double), cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(d_cfl, &cfl, sizeof(double), cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(d_nu, &nu, sizeof(double), cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(d_tau, &tau, sizeof(double), cudaMemcpyHostToDevice));
}


__global__ void boundaryCondition(const int nbrOfGrids,
	double *d_u1, double *d_u2, double *d_u3) {
	d_u1[0] = d_u1[1];
	d_u2[0] = -d_u2[1];
	d_u3[0] = d_u3[1];
	d_u1[nbrOfGrids - 1] = d_u1[nbrOfGrids - 2];
	d_u2[nbrOfGrids - 1] = -d_u2[nbrOfGrids - 2];
	d_u3[nbrOfGrids - 1] = d_u3[nbrOfGrids - 2];
}

__global__ void updateTau(const int nbrOfGrids, const double *d_u1,
	const double *d_u2, const double *d_u3, const double *d_gama,
	double *d_cMax, const double *d_h, const double *d_cfl, double *d_tau) {
	updateCMax(nbrOfGrids, d_u1, d_u2, d_u3, d_gama, d_cMax);
	*d_tau = *d_cfl * *d_h / *d_cMax;
}

// used in laxWendroffStep
__device__ void d_boundaryCondition(const int nbrOfGrids,
	double *d_u1, double *d_u2, double *d_u3) {
	d_u1[0] = d_u1[1];
	d_u2[0] = -d_u2[1];
	d_u3[0] = d_u3[1];
	d_u1[nbrOfGrids - 1] = d_u1[nbrOfGrids - 2];
	d_u2[nbrOfGrids - 1] = -d_u2[nbrOfGrids - 2];
	d_u3[nbrOfGrids - 1] = d_u3[nbrOfGrids - 2];
}

// used in laxWendroffStep
__device__ void updateFlux(const int nbrOfGrids, const double *d_u1, const double *d_u2,
	const double *d_u3, double *d_f1, double *d_f2, double *d_f3, const double *d_gama) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x;
	double rho, m, e, p;
	for (int i = index; i < nbrOfGrids; i += stride) {
		rho = d_u1[i];
		m = d_u2[i];
		e = d_u3[i];
		p = (*d_gama - 1) * (e - m * m / rho / 2);
		d_f1[i] = m;
		d_f2[i] = m * m / rho + p;
		d_f3[i] = m / rho * (e + p);
	}
}

// used in laxWendroffStep
__device__ void halfStep(const int nbrOfGrids, const double *d_u1, const double *d_u2,
	const double *d_u3, double *d_u1Temp, double *d_u2Temp, double *d_u3Temp,
	const double *d_f1, const double *d_f2, const double *d_f3, const double *d_tau, const double *d_h) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < nbrOfGrids; i += stride) {
		if ((i > 0) && (i < nbrOfGrids - 1)) {
			d_u1Temp[i] = (d_u1[i + 1] + d_u1[i]) / 2 - *d_tau / 2 / *d_h * (d_f1[i + 1] - d_f1[i]);
			d_u2Temp[i] = (d_u2[i + 1] + d_u2[i]) / 2 - *d_tau / 2 / *d_h * (d_f2[i + 1] - d_f2[i]);
			d_u3Temp[i] = (d_u3[i + 1] + d_u3[i]) / 2 - *d_tau / 2 / *d_h * (d_f3[i + 1] - d_f3[i]);
		}
	}
}

// used in laxWendroffStep
__device__ void step(const int nbrOfGrids, const double *d_u1, const double *d_u2,
	const double *d_u3, double *d_u1Temp, double *d_u2Temp, double *d_u3Temp,
	const double *d_f1, const double *d_f2, const double *d_f3, const double *d_tau, const double *d_h) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < nbrOfGrids; i += stride) {
		if ((i > 0) && (i < nbrOfGrids - 1)) {
			d_u1Temp[i] = d_u1[i] - *d_tau / *d_h * (d_f1[i] - d_f1[i - 1]);
			d_u2Temp[i] = d_u2[i] - *d_tau / *d_h * (d_f2[i] - d_f2[i - 1]);
			d_u3Temp[i] = d_u3[i] - *d_tau / *d_h * (d_f3[i] - d_f3[i - 1]);
		}
	}
}

// used in laxWendroffStep
__device__ void updateU(const int nbrOfGrids, double *d_u1, double *d_u2,
	double *d_u3, const double *d_u1Temp, const double *d_u2Temp, const double *d_u3Temp) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < nbrOfGrids; i += stride) {
		if ((i > 0) && (i < nbrOfGrids - 1)) {
			d_u1[i] = d_u1Temp[i];
			d_u2[i] = d_u2Temp[i];
			d_u3[i] = d_u3Temp[i];
		}
	}
}

__global__	void laxWendroffStep(const int nbrOfGrids, double *d_u1, double *d_u2,
	double *d_u3, double *d_u1Temp, double *d_u2Temp, double *d_u3Temp,
	double *d_f1, double *d_f2, double *d_f3, const double *d_tau, const double *d_h, const double *d_gama) {
	updateFlux(nbrOfGrids, d_u1, d_u2, d_u3, d_f1, d_f2, d_f3, d_gama);
	halfStep(nbrOfGrids, d_u1, d_u2, d_u3, d_u1Temp, d_u2Temp, d_u3Temp, d_f1, d_f2, d_f3, d_tau, d_h);
	d_boundaryCondition(nbrOfGrids, d_u1Temp, d_u2Temp, d_u3Temp);
	updateFlux(nbrOfGrids, d_u1Temp, d_u2Temp, d_u3Temp, d_f1, d_f2, d_f3, d_gama);
	step(nbrOfGrids, d_u1, d_u2, d_u3, d_u1Temp, d_u2Temp, d_u3Temp, d_f1, d_f2, d_f3, d_tau, d_h);
	updateU(nbrOfGrids, d_u1, d_u2, d_u3, d_u1Temp, d_u2Temp, d_u3Temp);
}


__global__	void RoeStep(const int nbrOfGrids, double *d_u1, double *d_u2,
	double *d_u3, double *d_u1Temp, double *d_u2Temp, double *d_u3Temp,
	double *d_f1, double *d_f2, double *d_f3, const double *d_tau, const double *d_h, const double *d_gama) {
	;
}
