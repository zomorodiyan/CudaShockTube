#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class ShockTube
{
public:
	ShockTube() { ; }
	~ShockTube(){ ; }

	// Host Tests
	void HostTest01(); void HostTest02(); //void HostTest03();

	// Device Tests
	void DeviceTest01(); //void DeviceTest02(); //void DeviceTest03(); 

	// Allocate space for host copies of the variables
	void allocHostMemory();

	// Allocate space for device copies of the variables
	void allocDeviceMemory();

	// Assigns Sod's shock tube problem initial conditions to host memory
	void initHostMemory();
	
	// reflection boundary condition at the both ends of the shock tube
	void hostBoundaryCondition();

	// Calculate and return average values of u, v, p, e
	double* getAverages();

	// Free allocated space for host copies of the variables
	void freeHostMemory();

	// Free allocated space for device copies of the variables
	void freeDeviceMemory();

	// host data members
	int nbrOfGrids = 200;	// number of grid points (default = 200)
	double h, length, gama, cfl, nu, tau, step, uAverage, roAverage,
		eAverage, pAverage;
	double *u1, *u2, *u3, *f1, *f2, *f3, *vol;

	// device data members
	double *d_h, *d_length, *d_gama, *d_cfl, *d_nu, *d_tau, *d_step;
	double *d_u1, *d_u2, *d_u3, *d_f1, *d_f2, *d_f3, *d_vol, *d_cMax;
	int size = nbrOfGrids * sizeof(int);

	// Calculate and return cMax
	double cMax();

	// copy device data members to host data members
	void copyDeviceToHost(const int nbrOfGrids);
};


// Calculate d_cMax
__device__ void calcCMax(const int nbrOfGrids, const double *d_u1,
	const double *d_u2, const double *d_u3, const double *d_gamma,
	double *d_cMax);

// Assigns Sod's shock tube problem initial conditions to device memory
__global__ void initDeviceMemory(const int nbrOfGrids, double *d_u1,
	double *d_u2, double *d_u3, double *d_vol, double *d_h,
	double *d_length, double *d_gama, double *d_cfl, double *d_nu,
	double *d_tau, double *d_step, double *d_cMax);
