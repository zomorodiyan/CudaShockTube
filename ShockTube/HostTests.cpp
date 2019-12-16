
#include <iostream>
#include <string>
#include "ShockTube.cuh"

#define fail " \033[1;31m"
#define pass " \033[1;32m"
#define reset "\033[0m"


void ShockTube::HostTest01() {
	const std::string test = "HostMemoryAllocatedAndInitializedCorrectly";
	std::cout << __func__;
	nbrOfGrids = 10;
	allocHostMemory();
	initHostMemory();
	double* Averages = getAverages();
	freeHostMemory();
	/*/ std::cout << Averages[0] << "  " << Averages[1] << "  " << Averages[2] << "  " << Averages[3] << std::endl; /**/
	double eps = 1e-14;
	if ((abs(Averages[0] - 0.65) < eps)
		&& (abs(Averages[1] - 0) < eps)
		&& (abs(Averages[2] - 1.6) < eps)
		&& (abs(Averages[3] - 0.64) < eps))
		std::cout << pass << test << reset << std::endl;
	else
		std::cout << fail << test << reset << std::endl;
}

void ShockTube::HostTest02() {
	const std::string test = "HostBoundaryConditionWorksCorrectly";
	std::cout << __func__;
	nbrOfGrids = 10;
	allocHostMemory();
	u1[1] = u1[8] = 1; u1[0] = u1[9] = 0;
	u2[1] = u2[8] = -1; u2[0] = u2[9] = 0;
	u3[1] = u3[8] = 1; u3[0] = u3[9] = 0;
	hostBoundaryCondition();
	if((1 == u1[0]) && (1 == u1[9]) && (1 == u2[0])
		&& (1 == u2[9]) && (1 == u3[0]) && (1 == u3[9]))
		std::cout << pass << test << reset << std::endl;
	else
		std::cout << fail << test << reset << std::endl;
}
