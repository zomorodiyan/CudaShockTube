#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class ShockTube
{
public:
	ShockTube() { ; }
	~ShockTube(){ ; }

	// Host Tests and Solutions
	void HostTest01(); void HostTest02(); void HostTest03(); void HostTest04();
	void LaxHost(); void RoeHost();

	// Device Tests and Solutions
	void DeviceTest01(); void DeviceTest02(); void DeviceTest03(); void DeviceTest04(); void DeviceTest05();
	void LaxDevice(); void RoeDevice();

	void writeToFile(std::string fileName);

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
	double h, length, gama, cfl, /*nu,*/ tau, cMax, uAverage, roAverage,
		eAverage, pAverage, t;
	double *u1, *u2, *u3, *f1, *f2, *f3, *vol, *u1Temp, *u2Temp, *u3Temp;

	// device data members
	double *d_h, *d_length, *d_gama, *d_cfl, *d_nu, *d_tau, *d_t;
	double *d_u1, *d_u2, *d_u3, *d_f1, *d_f2, *d_f3, *d_vol, *d_cMax;
	int size = nbrOfGrids * sizeof(int);

	// just used in Lax-Wendroff step
	double *d_u1Temp, *d_u2Temp, *d_u3Temp;

	// just used in Roe and pike step
	double *w1,*w2,*w3,*w4, *fc1,*fc2,*fc3, *fr1,*fr2,*fr3, *fl1,*fl2,*fl3, *fludif1,*fludif2,*fludif3,
		*rsumr, *utilde, *htilde, *uvdif, *absvt, *ssc, *vsc,
		*eiglam1,*eiglam2,*eiglam3, *sgn1,*sgn2,*sgn3, *a1, *a2, *a3, *ac11,*ac12,*ac13, *ac21,*ac22,*ac23;
	int *isb1, *isb2, *isb3;

	// Calculate and update cMax
	void hostUpdateCMax();

	// Calculate and update tau
	void hostUpdateTau();

	// copy device data members to host data members
	void copyDeviceToHost(const int nbrOfGrids);

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
	double *d_u3, const double *d_vol, double *d_f1, double *d_f2, double *d_f3, 
	const double *d_tau, const double *d_h, const double *d_gama,
	double *w1,double *w2,double *w3,double *w4, double *fc1,double *fc2,double *fc3,
	double *fr1,double *fr2,double *fr3, double *fl1,double *fl2,double *fl3,
	double *fludif1,double *fludif2,double *fludif3,
	double *rsumr, double *utilde, double *htilde, double *uvdif, double *absvt, double *ssc, double *vsc,
	double *eiglam1,double *eiglam2,double *eiglam3, double *sgn1,double *sgn2,double *sgn3,
	int *isb1,int *isb2,int *isb3, double *a1,double *a2,double *a3,
	double *ac11,double *ac12,double *ac13, double *ac21,double *ac22,double *ac23);
