
#include <iostream>
#include <fstream>
#include <string>
#include "ShockTube.cuh"
#include "Cmath""
//#include <iomanip>      // std::setprecision (use like std::endl)
//#define coutPericision 30

#define eps 1e-14
// used to change text color on terminal(or commond prompt)

#define fail " \033[1;31m"
#define pass " \033[1;32m"
#define yellow " \033[1;33m"
#define blue " \033[1;34m"
#define reset "\033[0m"


void ShockTube::HostTest01() {
	const std::string test = "Memory Allocation And Initialization";
	std::cout << yellow << __func__ << reset;
	nbrOfGrids = 10;
	allocHostMemory();
	initHostMemory();
	updateAverages();
	if ((abs(roAverage - 0.5625) < eps)
		&& (abs(uAverage - 0) < eps)
		&& (abs(eAverage - 1.375) < eps)
		&& (abs(pAverage - 0.55) < eps))
		std::cout << pass << test << reset << std::endl;
	else
		std::cout << fail << test << reset << std::endl;
	freeHostMemory();
}

void ShockTube::HostTest02() {
	const std::string test = "Boundary Condition";
	std::cout << yellow << __func__ << reset;
	nbrOfGrids = 10;
	allocHostMemory();
	u1[1] = u1[8] = 1; 	u2[1] = u2[8] = -1;	u3[1] = u3[8] = 1; 
	hostBoundaryCondition();
	if((1 == u1[0]) && (1 == u1[9]) && (1 == u2[0])
		&& (1 == u2[9]) && (1 == u3[0]) && (1 == u3[9]))
		std::cout << pass << test << reset << std::endl;
	else
		std::cout << fail << test << reset << std::endl;
	freeHostMemory();
}

void ShockTube::HostTest03() {
	const std::string test = "laxWendroff Step";
	std::cout << yellow << __func__ << reset;
	nbrOfGrids = 10;
	allocHostMemory();
	initHostMemory();
	hostLaxWendroffStep();
	if((abs(u1[4] - 0.739642857142857) < eps) && (abs(u2[4] - 0.21554331167307) < eps)
		&& (abs(u3[4] - 1.62828130612245) < eps) && (abs(u1[5] - 0.385357142857143) < eps)
		&& (abs(u2[5] - 0.46903163465702) < eps) && (abs(u3[5] - 1.1217186938775515) < eps))
		std::cout << pass << test << reset << std::endl;
	else
		std::cout << fail << test << reset << std::endl;
	freeHostMemory();
}

void ShockTube::HostTest04() {
	const std::string test = "Roe Step";
	std::cout << yellow << __func__ << reset;
	nbrOfGrids = 10;
	allocHostMemory();
	initHostMemory();
	hostRoeStep();
	if((abs(u1[4] - 0.702848465455315) < eps) && (abs(u2[4] - 0.342287473165049) < eps)
		&& (abs(u3[4] - 1.5143016216857514) < eps) && (abs(u1[5] - 0.422151534544684) < eps)
		&& (abs(u2[5] - 0.342287473165049) < eps) && (abs(u3[5] - 1.235698378314249) < eps))
		std::cout << pass << test << reset << std::endl;
	else
		std::cout << fail << test << reset << std::endl;
	freeHostMemory();
}

void ShockTube::LaxHost() {
	std::cout << yellow << __func__ << reset;
	nbrOfGrids = 256;
	allocHostMemory();
	initHostMemory();
	for (double tMax = 0.2, t = 0; t - tMax < -eps; t += tau)
	{
		hostBoundaryCondition();
		hostUpdateTau();
		if (t + tau - tMax > eps)
			tau = tMax - t;
		hostLaxWendroffStep();
	}
	writeToFile("LaxHost.dat");
	std::cout << blue << "solution: LaxHost.dat" << reset << std::endl;
	freeHostMemory();
}

void ShockTube::RoeHost() {
	std::cout << yellow << __func__ << reset;
	nbrOfGrids = 256;
	allocHostMemory();
	initHostMemory();
	hostBoundaryCondition();
	t = 0;
	for (double tMax = 0.2, t = 0; t - tMax < -eps; t += tau)
	{
		hostUpdateTau();
		if (t + tau > tMax)
			tau = tMax - t;
		hostRoeStep();
		hostBoundaryCondition();
	}
	writeToFile("RoeHost.dat");
	std::cout << blue << "solution: RoeHost.dat" << reset << std::endl;
	freeHostMemory();
}
