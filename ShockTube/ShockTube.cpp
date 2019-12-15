#include <iostream>
#include "ShockTube.cuh"

void ShockTube::logcpp(){
	std::cout << "I am in ShockTube.cpp" << std::endl;
}


// Allocate space for host copies of the variables
void ShockTube::allocHostMemory() {
	int size = nbrOfGrids * sizeof(double);
	u1 = (double*)malloc(size);
	u2 = (double*)malloc(size);
	u3 = (double*)malloc(size);
	f1 = (double*)malloc(size);
	f2 = (double*)malloc(size);
	f3 = (double*)malloc(size);
	vol = (double*)malloc(size);
}

// Calculate cMax
double ShockTube::cMax() {
	double ro, u, p, c, cMax = 0;
	for (int i = 0; i < nbrOfGrids; i++) {
		if (u1[i] == 0)
			continue;
		ro = u1[i];
		u = u2[i] / ro;
		p = (u3[i] - ro * u * u / 2) * (gama - 1);
		c = sqrt(gama * abs(p) / ro);
		if (cMax < c + abs(u))
			cMax = c + abs(u);
	}
	return cMax;
}

void ShockTube::initialize() {
	length = 1;                      // length of shock tube
	gama = 1.4;                 // ratio of specific heats
	cfl = 0.9;                  // Courant-Friedrichs-Lewy number
	nu = 0.0;                   // artificial viscosity coefficient
	h = length / (nbrOfGrids - 1);   // space grid size
	double ro, p, u = 0;
	for (int i = 0; i < nbrOfGrids; i++) {
		if (i > int(nbrOfGrids / 2)) { ro = 0.125, p = 0.1; }
		else { ro = 1, p = 1; }
		double e = p / (gama - 1) + ro * u * u / 2;
		u1[i] = ro;
		u2[i] = ro * u;
		u3[i] = e;
		vol[i] = 1;
	}
	tau = cfl * h / cMax();     // time grid size
	step = 0;
}

double* ShockTube::getAverages() {
	roAverage = uAverage = eAverage = pAverage = 0;
	double ro, u, e, p;
	for (int i = 0; i < nbrOfGrids; i++) {
		ro = u1[i];
		u = u2[i] / u1[i];
		e = u3[i];
		p = (u3[i] - u2[i] * u2[i] / u1[i] / 2) * (gama - 1);
		roAverage += ro;
		uAverage += u;
		eAverage += e;
		pAverage += p;
	}
	roAverage /= nbrOfGrids;
	uAverage /= nbrOfGrids;
	eAverage /= nbrOfGrids;
	pAverage /= nbrOfGrids;
	static double averages[4] = { roAverage, uAverage, eAverage, pAverage };
	return averages;
}
