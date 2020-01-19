
#include <iostream>
#include <fstream>
#include <string>
#include "ShockTube.cuh"
#include <iomanip>      // std::setprecision

#define coutPericision 30
#define eps 1e-14
#define fail " \033[1;31m"
#define pass " \033[1;32m"
#define yellow " \033[1;33m"
#define blue " \033[1;34m"
#define reset "\033[0m"
#define cudaErrorCheck(call)                                \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("\n%s (%d): %s%s%s\n", __FILE__, __LINE__, fail, err_str, reset);\
  }                                                         \
}
// Wrap device CUDA calls with cucheck_err as in the following example:
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
	boundaryCondition<<<1,16>>>(nbrOfGrids, d_u1, d_u2, d_u3);
	copyDeviceToHost(nbrOfGrids);
	freeDeviceMemory();
	if((1 == u1[0]) && (1 == u1[9]) && (1 == u2[0])
		&& (1 == u2[9]) && (1 == u3[0]) && (1 == u3[9]))
		std::cout << pass << test << reset << std::endl;
	else
		std::cout << fail << test << reset << std::endl;
	freeHostMemory();
}

void ShockTube::DeviceTest03() {
	const std::string test = "LaxWendroff Step";
	std::cout << yellow << __func__ << reset;
	nbrOfGrids = 10;
	allocDeviceMemory();
	initDeviceMemory<<<1,16>>>(nbrOfGrids, d_u1, d_u2, d_u3, d_vol, d_h, d_length, d_gama, d_cfl, d_nu, d_tau, d_cMax, d_t);
	laxWendroffStep<<<1,16>>>(nbrOfGrids, d_u1, d_u2, d_u3, d_u1Temp, d_u2Temp, d_u3Temp, 
		d_f1, d_f2, d_f3, d_tau, d_h, d_gama);
	allocHostMemory();
	copyDeviceToHost(nbrOfGrids);
	freeDeviceMemory();
	if((abs(u1[4] - 0.739642857142857) < eps) && (abs(u2[4] - 0.21554331167307) < eps)
		&& (abs(u3[4] - 1.62828130612245) < eps) && (abs(u1[5] - 0.385357142857143) < eps)
		&& (abs(u2[5] - 0.46903163465702) < eps) && (abs(u3[5] - 1.1217186938775515) < eps))
		std::cout << pass << test << reset << std::endl;
	else
		std::cout << fail << test << reset << std::endl;
	freeHostMemory();
}

void ShockTube::DeviceTest04() {
	const std::string test = "Roe Step";
	std::cout << yellow << __func__ << reset;
	nbrOfGrids = 10;
	allocDeviceMemory();
	initDeviceMemory<<<1,16>>>(nbrOfGrids, d_u1, d_u2, d_u3, d_vol, d_h, d_length, d_gama, d_cfl, d_nu, d_tau, d_cMax, d_t);
	RoeStep<<<1,16>>>(nbrOfGrids, d_u1, d_u2, d_u3, d_vol, d_f1, d_f2, d_f3, d_tau, d_h, d_gama,
	w1,w2,w3,w4, fc1,fc2,fc3, fr1,fr2,fr3, fl1,fl2,fl3, fludif1,fludif2,fludif3,
	rsumr, utilde, htilde, uvdif, absvt, ssc, vsc,
	eiglam1,eiglam2,eiglam3, sgn1,sgn2,sgn3, isb1,isb2,isb3, a1,a2,a3, ac11,ac12,ac13, ac21,ac22,ac23);
	allocHostMemory();
	copyDeviceToHost(nbrOfGrids);
	freeDeviceMemory();
	if((abs(u1[4] - 0.702848465455315) < eps) && (abs(u2[4] - 0.342287473165049) < eps)
		&& (abs(u3[4] - 1.5143016216857514) < eps) && (abs(u1[5] - 0.422151534544684) < eps)
		&& (abs(u2[5] - 0.342287473165049) < eps) && (abs(u3[5] - 1.235698378314249) < eps))
		std::cout << pass << test << reset << std::endl;
	else
		std::cout << fail << test << reset << std::endl;
	freeHostMemory();
}

void ShockTube::LaxDevice() {
	std::cout << yellow << __func__ << reset;
	nbrOfGrids = 101;
	allocDeviceMemory();
	initDeviceMemory<<<1,16>>>(nbrOfGrids, d_u1, d_u2, d_u3, d_vol, d_h, d_length, d_gama, d_cfl, d_nu, d_tau, d_cMax, d_t);
	allocHostMemory();
	double tMax = 0.2; t = 0;

	// decrease tau to not overshoot tMax 
	cudaErrorCheck(cudaMemcpy(&tau, d_tau, sizeof(double), cudaMemcpyDeviceToHost));
	if (tau - tMax > eps) 
		tau = tMax;
	cudaErrorCheck(cudaMemcpy(d_tau, &tau, sizeof(double), cudaMemcpyHostToDevice));
	int step = 1;
	for(bool tMaxReached = false; tMaxReached==false; step++)
	{
		boundaryCondition<<<1,16>>>(nbrOfGrids, d_u1, d_u2, d_u3);
		updateTau<<<1,1>>>(nbrOfGrids, d_u1, d_u2, d_u3, d_gama, d_cMax, d_h, d_cfl, d_tau); 

		// decrease tau to not overshoot tMax
		cudaErrorCheck(cudaMemcpy(&tau, d_tau, sizeof(double), cudaMemcpyDeviceToHost));
		if (t + tau - tMax > -eps)
		{ 
			tau = tMax - t;
			tMaxReached = true;
		} 
		cudaErrorCheck(cudaMemcpy(d_tau, &tau, sizeof(double), cudaMemcpyHostToDevice));

		laxWendroffStep<<<1,16>>>(nbrOfGrids, d_u1, d_u2, d_u3, d_u1Temp, d_u2Temp, d_u3Temp, 
			d_f1, d_f2, d_f3, d_tau, d_h, d_gama);
		t += tau;
	}
	copyDeviceToHost(nbrOfGrids);
	freeDeviceMemory();
	std::ofstream myfile;
	myfile.open("LaxDevice.dat");
	myfile << "variables = x, rho, u, p, mo, e, et, T, c, M, h" << std::endl;
	for (int i = 0; i < nbrOfGrids; i++) {
		double rho = u1[i];
		double u = u2[i] / rho;
		double p = (u3[i] - rho * u * u / 2) * (gama - 1);
		double m = u2[i]; // Momentum I think(?)
		double e = u3[i];
		//double e = p / (gama - 1) / rho; // is this line equivalent to the previous?
		double E = p / (gama - 1.) + 0.5 * rho * u * u;
		double T = p / rho;
		double c = sqrt(gama * p / rho);
		double M = u / c;
		double h = e + p / rho;
		double x = double(i) / double(nbrOfGrids);
		myfile << x << " " << rho << " " << u << " " << p << " " << m << " " << e << " " << E 
			<< " " << T << " " << c << " " << M << " " << h << "\n";
	}
	myfile.close();
	std::cout << blue << "solution: LaxDevice.dat" << reset << std::endl;
	freeHostMemory();
}
