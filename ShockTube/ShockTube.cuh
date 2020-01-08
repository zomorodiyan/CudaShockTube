#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class ShockTube
{
public:
	ShockTube() { ; }
	~ShockTube(){ ; }

	// Host Tests
	void HostTest01(); void HostTest02(); void HostTest03(); void HostTest04();

	// Device Tests
	void DeviceTest01(); void DeviceTest02(); void DeviceTest03(); void DeviceTest04();

	// Allocate space for host copies of the variables
	void allocHostMemory();

	// Allocate space for device copies of the variables
	void allocDeviceMemory();

	// Assigns Sod's shock tube problem initial conditions to host memory
	void initHostMemory();
	
	// reflection boundary condition at both ends of the tube
	void hostBoundaryCondition();

	// reflection boundary condition at both ends of the tube
	void hostBoundaryConditionTemp();

	// Calculate and return average values of u, v, p, e
	void updateAverages();

	// Free allocated space for host copies of the variables
	void freeHostMemory();

	// Free allocated space for device copies of the variables
	void freeDeviceMemory();

	// host data members
	int nbrOfGrids = 200;	// number of grid points (default = 200)
	double h, length, gama, cfl, nu, tau, cMax, uAverage, roAverage,
		eAverage, pAverage, t;
	double *u1, *u2, *u3, *f1, *f2, *f3, *vol, *u1Temp, *u2Temp, *u3Temp;

	// device data members
	double *d_h, *d_length, *d_gama, *d_cfl, *d_nu, *d_tau, *d_t;
	double *d_u1, *d_u2, *d_u3, *d_f1, *d_f2, *d_f3, *d_vol, *d_cMax, *d_u1Temp, *d_u2Temp, *d_u3Temp;
	int size = nbrOfGrids * sizeof(int);

	// Calculate and update cMax
	void hostUpdateCMax();

	// Calculate and update tau
	void hostUpdateTau();

	// copy device data members to host data members
	void copyDeviceToHost(const int nbrOfGrids);

	// copy flux from device to host (for debegging purpose)
	void copyFluxFromDeviceToHost(const int nbrOfGrids);

	// copy host data members to device data members
	void copyHostToDevice(const int nbrOfGrids);

// used in laxWendroffStep
	void updateFlux();

// used in laxWendroffStep
	void updateFluxTemp();

// used in laxWendroffStep
	void halfStep();

// used in laxWendroffStep
	void step();

// used in laxWendroffStep
	void updateU();

	void hostLaxWendroffStep();

	void hostRoeStep();

	void lapidusViscosity();
};


// Calculate d_cMax
__device__ void updateCMax(const int nbrOfGrids, const double *d_u1,
	const double *d_u2, const double *d_u3, const double *d_gamma,
	double *d_cMax);

// Assigns Sod's shock tube problem initial conditions to device memory
__global__ void initDeviceMemory(const int nbrOfGrids, double *d_u1,
	double *d_u2, double *d_u3, double *d_vol, double *d_h,
	double *d_length, double *d_gama, double *d_cfl, double *d_nu,
	double *d_tau, double *d_cMax, double *d_t);

__global__ void boundaryCondition(const int nbrOfGrids,
	double *d_u1, double *d_u2, double *d_u3);

__device__ void d_boundaryCondition(const int nbrOfGrids,
	double *d_u1, double *d_u2, double *d_u3);

__global__ void updateTau(const int nbrOfGrids, const double *d_u1,
	const double *d_u2, const double *d_u3, const double *d_gama,
	double *d_cMax, const double *d_h, const double *d_cfl, double *d_tau);

// used in laxWendroffStep
__device__ void updateFlux(const int nbrOfGrids, const double *d_u1, const double *d_u2,
		const double *d_u3, double *d_f1, double *d_f2, double *d_f3, const double *d_gama);

// used in laxWendroffStep
__device__ void halfStep(const int nbrOfGrids, const double *d_u1, const double *d_u2,
	const double *d_u3, double *d_u1Temp, double *d_u2Temp, double *d_u3Temp,
	const double *d_f1, const double *d_f2, const double *d_f3, const double *d_tau, const double *d_h);

// used in laxWendroffStep
__device__ void step(const int nbrOfGrids, const double *d_u1, const double *d_u2,
	const double *d_u3, double *d_u1Temp, double *d_u2Temp, double *d_u3Temp,
	const double *d_f1, const double *d_f2, const double *d_f3, const double *d_tau, const double *d_h);

// used in laxWendroffStep
__device__ void updateU(const int nbrOfGrids, double *d_u1, double *d_u2,
	double *d_u3, const double *d_u1Temp, const double *d_u2Temp, const double *d_u3Temp);

__global__	void laxWendroffStep(const int nbrOfGrids, double *d_u1, double *d_u2,
	double *d_u3, double *d_u1Temp, double *d_u2Temp, double *d_u3Temp,
	double *d_f1, double *d_f2, double *d_f3, const double *d_tau, const double *d_h, const double *d_gama);

__global__	void RoeStep(const int nbrOfGrids, double *d_u1, double *d_u2,
	double *d_u3, double *d_u1Temp, double *d_u2Temp, double *d_u3Temp,
	double *d_f1, double *d_f2, double *d_f3, const double *d_tau, const double *d_h, const double *d_gama);
