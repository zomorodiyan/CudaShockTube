
#include <iostream>
#include <fstream>
#include <string>
#include "ShockTube.cuh"
#include "Cmath""
#include <iomanip>      // std::setprecision

#define coutPericision 30
#define eps 1e-14
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
	/*/ std::cout << roAverage << "  " << uAverage << "  " << eAverage << "  " << pAverage << std::endl; /**/
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
	nbrOfGrids = 101;
	allocHostMemory();
	initHostMemory();
	t = 0;
	const double tMax = 0.2;
	// decrease tau to not overshoot tMax 
	if (tau - tMax > eps) 
		tau = tMax;
	int step = 1;
	for(bool tMaxReached = false; tMaxReached == false; step++)
	{
		hostBoundaryCondition();
		hostUpdateTau();
		if (t + tau - tMax >  -eps) {
			tau = tMax - t;
			tMaxReached = true;
		}
		hostLaxWendroffStep();
		t += tau;
	}
	std::ofstream myfile;
	myfile.open("LaxHost.dat");
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
	std::cout << blue << "solution: LaxHost.dat" << reset << std::endl;
	freeHostMemory();
}

void ShockTube::RoeHost() {
	std::cout << yellow << __func__ << reset;
	nbrOfGrids = 101;
	allocHostMemory();
	initHostMemory();
	hostBoundaryCondition();
	/*/
	std::cout << std::endl << " u1[0]: " << u1[0] << " u2[0]: " << u2[0] << " u3[0]: " << u3[0] <<
		" u1[1]: " << u1[1] << " u2[1]: " << u2[1] << " u3[1]: " << u3[1];
	std::cout << std::endl << " u1[2]: " << u1[2] << " u2[2]: " << u2[2] << " u3[2]: " << u3[2] <<
		" u1[3]: " << u1[3] << " u2[3]: " << u2[3] << " u3[3]: " << u3[3];
	std::cout << std::endl << " u1[4]: " << u1[4] << " u2[4]: " << u2[4] << " u3[4]: " << u3[4] <<
		" u1[5]: " << u1[5] << " u2[5]: " << u2[9] << " u3[5]: " << u3[5];
	std::cout << std::endl << " u1[6]: " << u1[6] << " u2[6]: " << u2[6] << " u3[6]: " << u3[8] <<
		" u1[7]: " << u1[7] << " u2[7]: " << u2[7] << " u3[7]: " << u3[7];
	std::cout << std::endl << " u1[8]: " << u1[8] << " u2[8]: " << u2[8] << " u3[8]: " << u3[8] <<
		" u1[9]: " << u1[9] << " u2[9]: " << u2[9] << " u3[9]: " << u3[9] << std::endl;
	/**/
	t = 0;
	double tMax = 0.2;
	int step = 1;
	for(bool tMaxReached = false; !tMaxReached; step++)
	{
		hostRoeStep();
		hostBoundaryCondition();

		if (step > 1)
		{
	std::cout << std::endl << " u1[0]: " << u1[0] << " u2[0]: " << u2[0] << " u3[0]: " << u3[0] <<
		" u1[1]: " << u1[1] << " u2[1]: " << u2[1] << " u3[1]: " << u3[1];
	std::cout << std::endl << " u1[2]: " << u1[2] << " u2[2]: " << u2[2] << " u3[2]: " << u3[2] <<
		" u1[3]: " << u1[3] << " u2[3]: " << u2[3] << " u3[3]: " << u3[3];
	std::cout << std::endl << " u1[4]: " << u1[4] << " u2[4]: " << u2[4] << " u3[4]: " << u3[4] <<
		" u1[5]: " << u1[5] << " u2[5]: " << u2[9] << " u3[5]: " << u3[5];
	std::cout << std::endl << " u1[6]: " << u1[6] << " u2[6]: " << u2[6] << " u3[6]: " << u3[8] <<
		" u1[7]: " << u1[7] << " u2[7]: " << u2[7] << " u3[7]: " << u3[7];
	std::cout << std::endl << " u1[8]: " << u1[8] << " u2[8]: " << u2[8] << " u3[8]: " << u3[8] <<
		" u1[9]: " << u1[9] << " u2[9]: " << u2[9] << " u3[9]: " << u3[9] << std::endl;
			/*/
			if ((abs(u1[4] - 0.702848465455315) < eps) && (abs(u2[4] - 0.342287473165049) < eps)
				&& (abs(u3[4] - 1.5143016216857514) < eps) && (abs(u1[5] - 0.422151534544684) < eps)
				&& (abs(u2[5] - 0.342287473165049) < eps) && (abs(u3[5] - 1.235698378314249) < eps))
				std::cout << pass << "Host first step solution " << reset;
			else
				std::cout << fail << "Host first step solution " << reset;
			/**/
		}

		t += tau;
		hostUpdateTau();
		// decrease tau to not overshoot tMax (works for tMax >= tau(initial value))
		if (t + tau > tMax){
			tau = tMax - t;
			tMaxReached = true;
		}
		if (step > 1)
			std::cout << "t: " << t  << ", tau: " << tau << " | ";
	}
	std::ofstream myfile;
	myfile.open("RoeHost.dat");
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
	std::cout << blue << "solution: RoeHost.dat" << reset << std::endl;
	freeHostMemory();
}
