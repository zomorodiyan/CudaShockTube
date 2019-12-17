#include <iostream>
#include "ShockTube.cuh"


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
	u1Temp = (double*)malloc(size);
	u2Temp = (double*)malloc(size);
	u3Temp = (double*)malloc(size);
}

// Calculate and update value of cMax
void ShockTube::updateCMax() {
	double ro, u, p, c; cMax = 0;
	for (int i = 0; i < nbrOfGrids; i++) {
		if (u1[i] == 0)
			continue;
		ro = u1[i];
		u = u2[i] / ro;
		p = (u3[i] - ro * u * u / 2) * (gama - 1);
		c = sqrt(gama * abs(p) / ro);
		if (cMax < c + abs(u)){
			cMax = c + abs(u);
		}
	}
}

void ShockTube::initHostMemory() {
	t = 0;						// time
	length = 1;                 // length of shock tube
	gama = 1.4;                 // ratio of specific heats
	cfl = 0.9;                  // Courant-Friedrichs-Lewy number
	nu = 0.0;                   // artificial viscosity coefficient
	h = length / (nbrOfGrids - 1);   // space grid size
	double ro, p, u = 0;
	for (int i = 0; i < nbrOfGrids; i++) {
		if (i >= int(nbrOfGrids / 2)) { ro = 0.125, p = 0.1; }
		else { ro = 1, p = 1; }
		double e = p / (gama - 1) + ro * u * u / 2;
		u1[i] = ro;
		u2[i] = ro * u;
		u3[i] = e;
		vol[i] = 1;
	}
	updateCMax();
	tau = cfl * h / cMax;     // time grid size
}

void ShockTube::updateAverages() {
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
}

// reflection boundary condition at both ends of the tube
void ShockTube::hostBoundaryCondition() {
	u1[0] = u1[1];
	u2[0] = -u2[1];
	u3[0] = u3[1];
	u1[nbrOfGrids - 1] = u1[nbrOfGrids - 2];
	u2[nbrOfGrids - 1] = -u2[nbrOfGrids - 2];
	u3[nbrOfGrids - 1] = u3[nbrOfGrids - 2];
}

// reflection boundary condition at both ends of the tube (Temp)
void ShockTube::hostBoundaryConditionTemp() {
	u1Temp[0] = u1Temp[1];
	u2Temp[0] = -u2Temp[1];
	u3Temp[0] = u3Temp[1];
	u1Temp[nbrOfGrids - 1] = u1Temp[nbrOfGrids - 2];
	u2Temp[nbrOfGrids - 1] = -u2Temp[nbrOfGrids - 2];
	u3Temp[nbrOfGrids - 1] = u3Temp[nbrOfGrids - 2];
}

// Free allocated space for host copies of the variables
void ShockTube::freeHostMemory() {
	free(u1);
	free(u2);
	free(u3);
	free(f1);
	free(f2);
	free(f3);
	free(vol);
}

// Calculate and return tau
void ShockTube::updateTau() {
	updateCMax();
	tau = cfl * h / cMax;
}

void ShockTube::laxWendroffStep(){
	// compute flux F from U
	for (int j = 0; j < nbrOfGrids; j++) {
		double rho = u1[j];
		double m = u2[j];
		double e = u3[j];
		double p = (gama - 1) * (e - m * m / rho / 2);
		f1[j] = m;
		f2[j] = m * m / rho + p;
		f3[j] = m / rho * (e + p);
	}

	// half step
	for (int j = 1; j < nbrOfGrids - 1; j++){
			u1Temp[j] = (u1[j + 1] + u1[j]) / 2 - tau / 2 / h * (f1[j + 1] - f1[j]);
			u2Temp[j] = (u2[j + 1] + u2[j]) / 2 - tau / 2 / h * (f2[j + 1] - f2[j]);
			u3Temp[j] = (u3[j + 1] + u3[j]) / 2 - tau / 2 / h * (f3[j + 1] - f3[j]);
	}

	hostBoundaryConditionTemp();

	// compute flux at half steps
	for (int j = 0; j < nbrOfGrids; j++) {
		double rho = u1Temp[j];
		double m = u2Temp[j];
		double e = u3Temp[j];
		double p = (gama - 1) * (e - m * m / rho / 2);
		f1[j] = m;
		f2[j] = m * m / rho + p;
		f3[j] = m / rho * (e + p);
	}

	// step using half step flux
	for (int j = 1; j < nbrOfGrids - 1; j++){
		u1Temp[j] = u1[j] - tau / h * (f1[j] - f1[j - 1]);
		u2Temp[j] = u2[j] - tau / h * (f2[j] - f2[j - 1]);
		u3Temp[j] = u3[j] - tau / h * (f3[j] - f3[j - 1]);
	}

	// update U from newU
	for (int j = 1; j < nbrOfGrids - 1; j++){
		u1[j] = u1Temp[j];
		u2[j] = u2Temp[j];
		u3[j] = u3Temp[j];
	}
}

void ShockTube::lapidusViscosity() {
	// store Delta_U values in newU
	for (int j = 1; j < nbrOfGrids; j++) {
		u1Temp[j] = u1[j] - u1[j - 1];
		u2Temp[j] = u2[j] - u2[j - 1];
		u3Temp[j] = u3[j] - u3[j - 1];
	}

	// multiply Delta_U by |Delta_U|
	for (int j = 1; j < nbrOfGrids; j++) {
		u1Temp[j] *= abs(u1Temp[j]);
		u2Temp[j] *= abs(u2Temp[j]);
		u3Temp[j] *= abs(u3Temp[j]);
	}

	// add artificial viscosity
	for (int j = 2; j < nbrOfGrids; j++) {
		u1[j] += nu * tau / h * (u1Temp[j] - u1Temp[j - 1]);
		u2[j] += nu * tau / h * (u2Temp[j] - u2Temp[j - 1]);
		u3[j] += nu * tau / h * (u3Temp[j] - u3Temp[j - 1]);
	}
}

