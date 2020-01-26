#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <assert.h>
#include <iostream>

#include "ShockTube.cuh"

//#include <algorithm> // in order to use std::max and std::min

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
		// only used in Lax-Wendroff step
	cudaErrorCheck(cudaMalloc((void **)&d_u1Temp, size));
	cudaErrorCheck(cudaMalloc((void **)&d_u2Temp, size));
	cudaErrorCheck(cudaMalloc((void **)&d_u3Temp, size));
		// only used in Roe step
	cudaErrorCheck(cudaMalloc((void **)&w1, size));
	cudaErrorCheck(cudaMalloc((void **)&w2, size));
	cudaErrorCheck(cudaMalloc((void **)&w3, size));
	cudaErrorCheck(cudaMalloc((void **)&w4, size));
	cudaErrorCheck(cudaMalloc((void **)&fc1, size));
	cudaErrorCheck(cudaMalloc((void **)&fc2, size));
	cudaErrorCheck(cudaMalloc((void **)&fc3, size));
	cudaErrorCheck(cudaMalloc((void **)&fr1, size));
	cudaErrorCheck(cudaMalloc((void **)&fr2, size));
	cudaErrorCheck(cudaMalloc((void **)&fr3, size));
	cudaErrorCheck(cudaMalloc((void **)&fl1, size));
	cudaErrorCheck(cudaMalloc((void **)&fl2, size));
	cudaErrorCheck(cudaMalloc((void **)&fl3, size));
	cudaErrorCheck(cudaMalloc((void **)&fludif1, size));
	cudaErrorCheck(cudaMalloc((void **)&fludif2, size));
	cudaErrorCheck(cudaMalloc((void **)&fludif3, size));
	cudaErrorCheck(cudaMalloc((void **)&eiglam1, size));
	cudaErrorCheck(cudaMalloc((void **)&eiglam2, size));
	cudaErrorCheck(cudaMalloc((void **)&eiglam3, size));
	cudaErrorCheck(cudaMalloc((void **)&sgn1, size));
	cudaErrorCheck(cudaMalloc((void **)&sgn2, size));
	cudaErrorCheck(cudaMalloc((void **)&sgn3, size));
	cudaErrorCheck(cudaMalloc((void **)&a1, size));
	cudaErrorCheck(cudaMalloc((void **)&a2, size));
	cudaErrorCheck(cudaMalloc((void **)&a3, size));
	cudaErrorCheck(cudaMalloc((void **)&ac11, size));
	cudaErrorCheck(cudaMalloc((void **)&ac12, size));
	cudaErrorCheck(cudaMalloc((void **)&ac13, size));
	cudaErrorCheck(cudaMalloc((void **)&ac21, size));
	cudaErrorCheck(cudaMalloc((void **)&ac22, size));
	cudaErrorCheck(cudaMalloc((void **)&ac23, size));
	cudaErrorCheck(cudaMalloc((void **)&rsumr, size));
	cudaErrorCheck(cudaMalloc((void **)&utilde, size));
	cudaErrorCheck(cudaMalloc((void **)&htilde, size));
	cudaErrorCheck(cudaMalloc((void **)&uvdif, size));
	cudaErrorCheck(cudaMalloc((void **)&absvt, size));
	cudaErrorCheck(cudaMalloc((void **)&ssc, size));
	cudaErrorCheck(cudaMalloc((void **)&vsc, size));
	cudaErrorCheck(cudaMalloc((void **)&isb1, nbrOfGrids * sizeof(int)));
	cudaErrorCheck(cudaMalloc((void **)&isb2, nbrOfGrids * sizeof(int)));
	cudaErrorCheck(cudaMalloc((void **)&isb3, nbrOfGrids * sizeof(int)));
}

// Free allocated space for device copies of the variables
void ShockTube::freeDeviceMemory() {
	cudaErrorCheck(cudaFree(d_u1));
	cudaErrorCheck(cudaFree(d_u2));
	cudaErrorCheck(cudaFree(d_u3));
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
		// only used in Lax-Wendroff step
	cudaErrorCheck(cudaFree(d_u1Temp));
	cudaErrorCheck(cudaFree(d_u2Temp));
	cudaErrorCheck(cudaFree(d_u3Temp));
		// only used in Roe step 
	cudaErrorCheck(cudaFree(w1));
	cudaErrorCheck(cudaFree(w2));
	cudaErrorCheck(cudaFree(w3));
	cudaErrorCheck(cudaFree(w4));
	cudaErrorCheck(cudaFree(fc1)); 
	cudaErrorCheck(cudaFree(fc2));
	cudaErrorCheck(cudaFree(fc3));
	cudaErrorCheck(cudaFree(fr1)); 
	cudaErrorCheck(cudaFree(fr2)); 
	cudaErrorCheck(cudaFree(fr3));
	cudaErrorCheck(cudaFree(fl1));
	cudaErrorCheck(cudaFree(fl2));
	cudaErrorCheck(cudaFree(fl3));
	cudaErrorCheck(cudaFree(fludif1)); 
	cudaErrorCheck(cudaFree(fludif2)); 
	cudaErrorCheck(cudaFree(fludif3));
	cudaErrorCheck(cudaFree(eiglam1));
	cudaErrorCheck(cudaFree(eiglam2)); 
	cudaErrorCheck(cudaFree(eiglam3));
	cudaErrorCheck(cudaFree(sgn1)); 
	cudaErrorCheck(cudaFree(sgn2)); 
	cudaErrorCheck(cudaFree(sgn3));
	cudaErrorCheck(cudaFree(isb1)); 
	cudaErrorCheck(cudaFree(isb2)); 
	cudaErrorCheck(cudaFree(isb3));
	cudaErrorCheck(cudaFree(a1)); 
	cudaErrorCheck(cudaFree(a2)); 
	cudaErrorCheck(cudaFree(a3));
	cudaErrorCheck(cudaFree(ac11)); 
	cudaErrorCheck(cudaFree(ac12)); 
	cudaErrorCheck(cudaFree(ac13));
	cudaErrorCheck(cudaFree(ac21)); 
	cudaErrorCheck(cudaFree(ac22)); 
	cudaErrorCheck(cudaFree(ac23));
	cudaErrorCheck(cudaFree(rsumr));
	cudaErrorCheck(cudaFree(utilde));
	cudaErrorCheck(cudaFree(htilde));
	cudaErrorCheck(cudaFree(uvdif));
	cudaErrorCheck(cudaFree(absvt));
	cudaErrorCheck(cudaFree(ssc));
	cudaErrorCheck(cudaFree(vsc));
}

// calculate and update value of d_cMax
__device__ void updateCMax(const int nbrOfGrids, const double *d_u1, 
	const double *d_u2, const double *d_u3, const double *d_gama, double *d_cMax) 
{ 
	*d_cMax = 0; int index = blockIdx.x * blockDim.x + threadIdx.x;
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
	*d_t = 0;								// time
	*d_length = 1;							// length of shock tube
	*d_gama = 1.4;							// ratio of specific heats
	*d_cfl = 0.9;							// Courant-Friedrichs-Lewy number
	*d_nu = 0.0;							// artificial viscosity coefficient
	*d_h = *d_length / (nbrOfGrids - 1);	// space grid size
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x;
	for(int i = index; i < nbrOfGrids; i+= stride){
		double e, ro, p, u = 0;
		if (i < nbrOfGrids){
			if (i >= int(nbrOfGrids / 2)) { ro = 0.125, p = 0.1; }
			else { ro = 1, p = 1; }
			e = p / (*d_gama - 1) + ro * u * u / 2;
			d_u1[i] = ro;
			d_u2[i] = ro * u;
			d_u3[i] = e;
			d_u3[i] = e;
			d_vol[i] = 1;
		}
	}
	updateCMax(nbrOfGrids, d_u1, d_u2, d_u3, d_gama, d_cMax); 
	*d_tau = (*d_cfl) * (*d_h) / (*d_cMax);    // initial time grid size, It will be modified to tMax if this > tMax
}

// copy device data members to host data members
void ShockTube::copyDeviceToHost(const int nbrOfGrids) {
	int size = nbrOfGrids * sizeof(double);
	cudaErrorCheck(cudaMemcpy(u1, d_u1, size, cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(u2, d_u2, size, cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaMemcpy(u3, d_u3, size, cudaMemcpyDeviceToHost));
}

// copy host data members to device data members
void ShockTube::copyHostToDevice(const int nbrOfGrids) {
	int size = nbrOfGrids * sizeof(double);
	cudaErrorCheck(cudaMemcpy(d_u1, u1, size, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(d_u2, u2, size, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(d_u3, u3, size, cudaMemcpyHostToDevice));
}

__global__ void updateTau(const int nbrOfGrids, const double *d_u1,
	const double *d_u2, const double *d_u3, const double *d_gama,
	double *d_cMax, const double *d_h, const double *d_cfl, double *d_tau) {
	updateCMax(nbrOfGrids, d_u1, d_u2, d_u3, d_gama, d_cMax);
	*d_tau = *d_cfl * *d_h / *d_cMax;
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
	d_boundaryCondition(nbrOfGrids, d_u1, d_u2, d_u3);
}

// used in RoeStep
	#define tiny 1e-30
	#define sbpar1 2.0
	#define sbpar2 2.0

__global__	void RoeStep(const int nbrOfGrids, double *d_u1, double *d_u2,
	double *d_u3, const double *d_vol, double *d_f1, double *d_f2, double *d_f3, 
	const double *d_tau, const double *d_h, const double *d_gama,
	double *w1,double *w2,double *w3,double *w4, double *fc1,double *fc2,double *fc3,
	double *fr1,double *fr2,double *fr3, double *fl1,double *fl2,double *fl3,
	double *fludif1,double *fludif2,double *fludif3,
	double *rsumr, double *utilde, double *htilde, double *uvdif, double *absvt, double *ssc, double *vsc,
	double *eiglam1,double *eiglam2,double *eiglam3, double *sgn1,double *sgn2,double *sgn3,
	int *isb1,int *isb2,int *isb3, double *a1,double *a2,double *a3,
	double *ac11,double *ac12,double *ac13, double *ac21,double *ac22,double *ac23) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < nbrOfGrids; i += stride) {

		// find parameter vector w
		{
			w1[i] = sqrt(d_vol[i] * d_u1[i]);
			w2[i] = w1[i] * d_u2[i] / d_u1[i];
			w4[i] = (*d_gama - 1) * (d_u3[i] - 0.5 * d_u2[i] * d_u2[i] / d_u1[i]);
			w3[i] = w1[i] * (d_u3[i] + w4[i]) / d_u1[i];
		}

		// calculate the fluxes at the cell center
		{
			fc1[i] = w1[i] * w2[i];
			fc2[i] = w2[i] * w2[i] + d_vol[i] * w4[i];
			fc3[i] = w2[i] * w3[i];
		}

		__syncthreads(); // because of the [i - 1] index below
		// calculate the fluxes at the cell walls 
		if (i > 0) {
			fl1[i] = fc1[i - 1]; fr1[i] = fc1[i];
			fl2[i] = fc2[i - 1]; fr2[i] = fc2[i];
			fl3[i] = fc3[i - 1]; fr3[i] = fc3[i];
		}

		// calculate the flux differences at the cell walls
		if (i > 0) {
			fludif1[i] = fr1[i] - fl1[i];
			fludif2[i] = fr2[i] - fl2[i];
			fludif3[i] = fr3[i] - fl3[i];
		}

		__syncthreads(); // because of the [i - 1] index below
		// calculate the tilded state variables = mean values at the interfaces
		if (i > 0) {
			rsumr[i] = 1 / (w1[i - 1] + w1[i]);

			utilde[i] = (w2[i - 1] + w2[i]) * rsumr[i];
			htilde[i] = (w3[i - 1] + w3[i]) * rsumr[i];

			absvt[i] = 0.5 * utilde[i] * utilde[i];
			uvdif[i] = utilde[i] * fludif2[i];

			ssc[i] = (*d_gama - 1) * (htilde[i] - absvt[i]);
			if (ssc[i] > 0.0)
				vsc[i] = sqrt(ssc[i]);
			else {
				vsc[i] = sqrt(abs(ssc[i]));
			}
		}

		// calculate the eigenvalues and projection coefficients for each eigenvector
		if (i > 0) {
			eiglam1[i] = utilde[i] - vsc[i];
			eiglam2[i] = utilde[i];
			eiglam3[i] = utilde[i] + vsc[i];
			sgn1[i] = eiglam1[i] < 0.0 ? -1 : 1;
			sgn2[i] = eiglam2[i] < 0.0 ? -1 : 1;
			sgn3[i] = eiglam3[i] < 0.0 ? -1 : 1;
			a1[i] = 0.5 * ((*d_gama - 1) * (absvt[i] * fludif1[i] + fludif3[i]
				- uvdif[i]) - vsc[i] * (fludif2[i] - utilde[i]
					* fludif1[i])) / ssc[i];
			a2[i] = (*d_gama - 1) * ((htilde[i] - 2 * absvt[i]) * fludif1[i]
				+ uvdif[i] - fludif3[i]) / ssc[i];
			a3[i] = 0.5 * ((*d_gama - 1) * (absvt[i] * fludif1[i] + fludif3[i]
				- uvdif[i]) + vsc[i] * (fludif2[i] - utilde[i]
					* fludif1[i])) / ssc[i];
		}

		// divide the projection coefficients by the wave speeds to evade expansion correction
		if (i > 0) {
			a1[i] /= eiglam1[i] + tiny;
			a2[i] /= eiglam2[i] + tiny;
			a3[i] /= eiglam3[i] + tiny;
		}

		// calculate the first order projection coefficients ac1
		if (i > 0) {
			ac11[i] = -sgn1[i] * a1[i] * eiglam1[i];
			ac12[i] = -sgn2[i] * a2[i] * eiglam2[i];
			ac13[i] = -sgn3[i] * a3[i] * eiglam3[i];
		}

		// apply the 'superbee' flux correction to made 2nd order projection coefficients ac2
		{
			ac21[1] = ac11[1];
			ac21[nbrOfGrids - 1] = ac11[nbrOfGrids - 1];
			ac22[1] = ac12[1];
			ac22[nbrOfGrids - 1] = ac12[nbrOfGrids - 1];
			ac23[1] = ac13[1];
			ac23[nbrOfGrids - 1] = ac13[nbrOfGrids - 1];


			double dtdx = *d_tau / *d_h;
			if ((i > 1) && (i < nbrOfGrids - 1)) {
				isb1[i] = i - int(sgn1[i]);
				ac21[i] = ac11[i] + eiglam1[i] *
					((fmax(0.0, fmin(sbpar1 * a1[isb1[i]], fmax(a1[i], fmin(a1[isb1[i]], sbpar2 * a1[i])))) +
						fmin(0.0, fmax(sbpar1 * a1[isb1[i]], fmin(a1[i], fmax(a1[isb1[i]], sbpar2 * a1[i]))))) *
						(sgn1[i] - dtdx * eiglam1[i]));
				isb2[i] = i - int(sgn2[i]);
				ac22[i] = ac12[i] + eiglam2[i] *
					((fmax(0.0, fmin(sbpar1 * a2[isb2[i]], fmax(a2[i], fmin(a2[isb2[i]], sbpar2 * a2[i])))) +
						fmin(0.0, fmax(sbpar1 * a2[isb2[i]], fmin(a2[i], fmax(a2[isb2[i]], sbpar2 * a2[i]))))) *
						(sgn2[i] - dtdx * eiglam2[i]));
				isb3[i] = i - int(sgn3[i]);
				ac23[i] = ac13[i] + eiglam3[i] *
					((fmax(0.0, fmin(sbpar1 * a3[isb3[i]], fmax(a3[i], fmin(a3[isb3[i]], sbpar2 * a3[i])))) +
						fmin(0.0, fmax(sbpar1 * a3[isb3[i]], fmin(a3[i], fmax(a3[isb3[i]], sbpar2 * a3[i]))))) *
						(sgn3[i] - dtdx * eiglam3[i]));
			}
		}

		// calculate the final fluxes
		if (i > 0) {
			d_f1[i] = 0.5 * (fl1[i] + fr1[i] + ac21[i] + ac22[i] + ac23[i]);
			d_f2[i] = 0.5 * (fl2[i] + fr2[i] + eiglam1[i] * ac21[i]
				+ eiglam2[i] * ac22[i] + eiglam3[i] * ac23[i]);
			d_f3[i] = 0.5 * (fl3[i] + fr3[i] + (htilde[i] - utilde[i] * vsc[i]) * ac21[i]
				+ absvt[i] * ac22[i] + (htilde[i] + utilde[i] * vsc[i]) * ac23[i]);
		}

		__syncthreads(); // because of the [i + 1] index below
		// update U
		if (i > 0 && i < nbrOfGrids - 1) {
			d_u1[i] -= *d_tau / *d_h * (d_f1[i + 1] - d_f1[i]);
			d_u2[i] -= *d_tau / *d_h * (d_f2[i + 1] - d_f2[i]);
			d_u3[i] -= *d_tau / *d_h * (d_f3[i + 1] - d_f3[i]);
		}

		d_boundaryCondition(nbrOfGrids, d_u1, d_u2, d_u3);
	}
}
