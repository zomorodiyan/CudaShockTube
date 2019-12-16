
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

