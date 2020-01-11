#include <iostream>
#include <algorithm> // in order to use std::max and std::min
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
void ShockTube::hostUpdateCMax() {
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
		if (i >= int(nbrOfGrids / 2)) { ro = 0.125; p = 0.1; }
		else { ro = 1; p = 1; }
		double e = p / (gama - 1) + ro * u * u / 2;
		u1[i] = ro;
		u2[i] = ro * u;
		u3[i] = e;
		vol[i] = 1;
	}
	hostUpdateCMax();
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
void ShockTube::hostUpdateTau() {
	hostUpdateCMax();
	tau = cfl * h / cMax;
}

// used in hostLaxWendroffStep
void ShockTube::updateFlux() {
	for (int j = 0; j < nbrOfGrids; j++) {
		double rho = u1[j];
		double m = u2[j];
		double e = u3[j];
		double p = (gama - 1) * (e - m * m / rho / 2);
		f1[j] = m;
		f2[j] = m * m / rho + p;
		f3[j] = m / rho * (e + p);
	}
}

// used in hostLaxWendroffStep
void ShockTube::updateFluxTemp() {
	for (int j = 0; j < nbrOfGrids; j++) {
		double rho = u1Temp[j];
		double m = u2Temp[j];
		double e = u3Temp[j];
		double p = (gama - 1) * (e - m * m / rho / 2);
		f1[j] = m;
		f2[j] = m * m / rho + p;
		f3[j] = m / rho * (e + p);
	}
}

// used in hostLaxWendroffStep
void ShockTube::halfStep() {
	for (int j = 1; j < nbrOfGrids - 1; j++){
			u1Temp[j] = (u1[j + 1] + u1[j]) / 2 - tau / 2 / h * (f1[j + 1] - f1[j]);
			u2Temp[j] = (u2[j + 1] + u2[j]) / 2 - tau / 2 / h * (f2[j + 1] - f2[j]);
			u3Temp[j] = (u3[j + 1] + u3[j]) / 2 - tau / 2 / h * (f3[j + 1] - f3[j]);
	}
}

// used in hostLaxWendroffStep
void ShockTube::step() {
	for (int j = 1; j < nbrOfGrids - 1; j++){
		u1Temp[j] = u1[j] - tau / h * (f1[j] - f1[j - 1]);
		u2Temp[j] = u2[j] - tau / h * (f2[j] - f2[j - 1]);
		u3Temp[j] = u3[j] - tau / h * (f3[j] - f3[j - 1]);
	}
}

// used in hostLaxWendroffStep
void ShockTube::updateU() {
	for (int j = 1; j < nbrOfGrids - 1; j++){
		u1[j] = u1Temp[j];
		u2[j] = u2Temp[j];
		u3[j] = u3Temp[j];
	}
}

void ShockTube::hostLaxWendroffStep(){
	updateFlux();
	halfStep();
	hostBoundaryConditionTemp();
	updateFluxTemp();
	step();
	updateU();
}

void ShockTube::hostRoeStep()
{
	const double tiny = 1e-30;
	const double sbpar1 = 2.0;
	const double sbpar2 = 2.0;

	// allocate temporary arrays
	double **fludif = new double*[nbrOfGrids];
	double *rsumr = new double[nbrOfGrids];
	double *utilde = new double[nbrOfGrids];
	double *htilde = new double[nbrOfGrids];
	double *absvt = new double[nbrOfGrids];
	double *uvdif = new double[nbrOfGrids];
	double *ssc = new double[nbrOfGrids];
	double *vsc = new double[nbrOfGrids];
	double **a = new double*[nbrOfGrids];
	double **ac1 = new double*[nbrOfGrids];
	double **ac2 = new double*[nbrOfGrids];
	double **w = new double*[nbrOfGrids];
	double **eiglam = new double*[nbrOfGrids];
	double **sgn = new double*[nbrOfGrids];
	double **fc = new double*[nbrOfGrids];
	double **fl = new double*[nbrOfGrids];
	double **fr = new double*[nbrOfGrids];
	double *ptest = new double[nbrOfGrids];
	int **isb = new int*[nbrOfGrids];
	for (int i = 0; i < nbrOfGrids; i++) {
		fludif[i] = new double[3];
		a[i] = new double[3];
		ac1[i] = new double[3];
		ac2[i] = new double[3];
		w[i] = new double[4];
		eiglam[i] = new double[3];
		sgn[i] = new double[3];
		fc[i] = new double[3];
		fl[i] = new double[3];
		fr[i] = new double[3];
		isb[i] = new int[3];
	}

	// find parameter vector w
	for (int i = 0; i <= nbrOfGrids - 1; i++) {
		w[i][0] = sqrt(vol[i] * u1[i]);
		w[i][1] = w[i][0] * u2[i] / u1[i];
		w[i][3] = (gama - 1) * (u3[i] - 0.5 * u2[i] * u2[i] / u1[i]);
		w[i][2] = w[i][0] * (u3[i] + w[i][3]) / u1[i];
	}
	// {{{
	// calculate the fluxes at the cell center
	for (int i = 0; i <= nbrOfGrids - 1; i++) {
		fc[i][0] = w[i][0] * w[i][1];
		fc[i][1] = w[i][1] * w[i][1] + vol[i] * w[i][3];
		fc[i][2] = w[i][1] * w[i][2];
	}

	// calculate the fes at the cell walls
	// assuming constant primitive variables
	for (int n = 0; n < 3; n++) {
		for (int i = 1; i <= nbrOfGrids - 1; i++) {
			fl[i][n] = fc[i - 1][n];
			fr[i][n] = fc[i][n];
		}
	}

	// calculate the flux differences at the cell walls
	for (int n = 0; n < 3; n++)
		for (int i = 1; i <= nbrOfGrids - 1; i++)
			fludif[i][n] = fr[i][n] - fl[i][n];

	// calculate the tilded state variables = mean values at the interfaces
	for (int i = 1; i <= nbrOfGrids - 1; i++) {
		rsumr[i] = 1 / (w[i - 1][0] + w[i][0]);

		utilde[i] = (w[i - 1][1] + w[i][1]) * rsumr[i];
		htilde[i] = (w[i - 1][2] + w[i][2]) * rsumr[i];

		absvt[i] = 0.5 * utilde[i] * utilde[i];
		uvdif[i] = utilde[i] * fludif[i][1];

		ssc[i] = (gama - 1) * (htilde[i] - absvt[i]);
		if (ssc[i] > 0.0)
			vsc[i] = sqrt(ssc[i]);
		else {
			vsc[i] = sqrt(abs(ssc[i]));
		}
	}

	// calculate the eigenvalues and projection coefficients for each
	// eigenvector
	for (int i = 1; i <= nbrOfGrids - 1; i++) {
		eiglam[i][0] = utilde[i] - vsc[i];
		eiglam[i][1] = utilde[i];
		eiglam[i][2] = utilde[i] + vsc[i];
		for (int n = 0; n < 3; n++)
			sgn[i][n] = eiglam[i][n] < 0.0 ? -1 : 1;
		a[i][0] = 0.5 * ((gama - 1) * (absvt[i] * fludif[i][0] + fludif[i][2]
			- uvdif[i]) - vsc[i] * (fludif[i][1] - utilde[i]
				* fludif[i][0])) / ssc[i];
		a[i][1] = (gama - 1) * ((htilde[i] - 2 * absvt[i]) * fludif[i][0]
			+ uvdif[i] - fludif[i][2]) / ssc[i];
		a[i][2] = 0.5 * ((gama - 1) * (absvt[i] * fludif[i][0] + fludif[i][2]
			- uvdif[i]) + vsc[i] * (fludif[i][1] - utilde[i]
				* fludif[i][0])) / ssc[i];
	}

	// divide the projection coefficients by the wave speeds
	// to evade expansion correction
	for (int n = 0; n < 3; n++)
		for (int i = 1; i <= nbrOfGrids - 1; i++)
			a[i][n] /= eiglam[i][n] + tiny;

	// calculate the first order projection coefficients ac1
	for (int n = 0; n < 3; n++)
		for (int i = 1; i <= nbrOfGrids - 1; i++)
			ac1[i][n] = -sgn[i][n] * a[i][n] * eiglam[i][n];

	// apply the 'superbee' flux correction to made 2nd order projection
	// coefficients ac2
	for (int n = 0; n < 3; n++) {
		ac2[1][n] = ac1[1][n];
		ac2[nbrOfGrids - 1][n] = ac1[nbrOfGrids - 1][n];
	}

	double dtdx = tau / h;
	for (int n = 0; n < 3; n++) {
		for (int i = 2; i <= nbrOfGrids - 2; i++) {
			isb[i][n] = i - int(sgn[i][n]);
			ac2[i][n] = ac1[i][n] + eiglam[i][n] *
				((std::max(0.0, std::min(sbpar1 * a[isb[i][n]][n], std::max(a[i][n],
					std::min(a[isb[i][n]][n], sbpar2 * a[i][n])))) +
					std::min(0.0, std::max(sbpar1 * a[isb[i][n]][n], std::min(a[i][n],
						std::max(a[isb[i][n]][n], sbpar2 * a[i][n]))))) *
						(sgn[i][n] - dtdx * eiglam[i][n]));
		}
	}

	// calculate the final fluxes
	for (int i = 1; i <= nbrOfGrids - 1; i++) {
		f1[i] = 0.5 * (fl[i][0] + fr[i][0] + ac2[i][0]
			+ ac2[i][1] + ac2[i][2]);
		f2[i] = 0.5 * (fl[i][1] + fr[i][1] +
			eiglam[i][0] * ac2[i][0] + eiglam[i][1] * ac2[i][1] +
			eiglam[i][2] * ac2[i][2]);
		f3[i] = 0.5 * (fl[i][2] + fr[i][2] +
			(htilde[i] - utilde[i] * vsc[i]) * ac2[i][0] +
			absvt[i] * ac2[i][1] +
			(htilde[i] + utilde[i] * vsc[i]) * ac2[i][2]);
	}
	// }}}
	/*/ {{{
	// calculate test variable for negative pressure check
	for (int i = 1; i <= nbrOfGrids - 2; i++) {
		ptest[i] = h * vol[i] * u2[i] +
			tau * (f2[i] - f2[i + 1]);
		ptest[i] = -ptest[i] * ptest[i] + 2 * (h * vol[i] * u1[i] +
			tau * (f1[i] - f1[i + 1])) * (h * vol[i] *
				u3[i] + tau * (f3[i] - f3[i + 1]));
	}

	// check for negative pressure/internal energy and set fluxes
	// left and  right to first order if detected
	for (int i = 1; i <= nbrOfGrids - 2; i++) {
		if (ptest[i] <= 0.0 || (h * vol[i] * u1[i] + tau * (f1[i]
			- f1[i + 1])) <= 0.0) {

			f1[i] = 0.5 * (fl[i][0] + fr[i][0] +
				ac1[i][0] + ac1[i][1] + ac1[i][2]);
			f2[i] = 0.5 * (fl[i][1] + fr[i][1] +
				eiglam[i][0] * ac1[i][0] + eiglam[i][1] * ac1[i][1] +
				eiglam[i][2] * ac1[i][2]);
			f3[i] = 0.5 * (fl[i][2] + fr[i][2] +
				(htilde[i] - utilde[i] * vsc[i]) * ac1[i][0] +
				absvt[i] * ac1[i][1] +
				(htilde[i] + utilde[i] * vsc[i]) * ac1[i][2]);
			f1[i + 1] = 0.5 * (fl[i + 1][0] + fr[i + 1][0] +
				ac1[i + 1][0] + ac1[i + 1][1] + ac1[i + 1][2]);
			f2[i + 1] = 0.5 * (fl[i + 1][1] + fr[i + 1][1] +
				eiglam[i + 1][0] * ac1[i + 1][0] + eiglam[i + 1][1] *
				ac1[i + 1][1] + eiglam[i + 1][2] * ac1[i + 1][2]);
			f3[i + 1] = 0.5 * (fl[i + 1][2] + fr[i + 1][2] +
				(htilde[i + 1] - utilde[i + 1] * vsc[i + 1]) * ac1[i + 1][0]
				+ absvt[i + 1] * ac1[i + 1][1] +
				(htilde[i + 1] + utilde[i + 1] * vsc[i + 1]) * ac1[i + 1][2]);

			// Check if it helped, set control variable if not

			ptest[i] = (h * vol[i] * u2[i] +
				tau * (f2[i] - f2[i + 1]));
			ptest[i] = 2.0 * (h * vol[i] * u1[i]
				+ tau * (f1[i] - f1[i + 1])) * (h * vol[i] *
					u3[i] + tau * (f3[i] - f3[i + 1]))
				- ptest[i] * ptest[i];
		}
	}
	/**/// }}}


	/*/
	// update U debug
	for (int j = 1; j < nbrOfGrids - 1; j++) {
		u1[j] = f1[j];
		u2[j] = f2[j];
		u3[j] = f3[j];
	}

	/*/
	// update U
	for (int j = 1; j < nbrOfGrids - 1; j++) {
		u1[j] -= tau / h * (f1[j + 1] - f1[j]);
		u2[j] -= tau / h * (f2[j + 1] - f2[j]);
		u3[j] -= tau / h * (f3[j + 1] - f3[j]);
	}
	/**/

	// free temporary arrays
	for (int i = 0; i < nbrOfGrids; i++) {
		delete[] fludif[i];
		delete[] a[i];
		delete[] ac1[i];
		delete[] ac2[i];
		delete[] w[i];
		delete[] eiglam[i];
		delete[] sgn[i];
		delete[] fc[i];
		delete[] fl[i];
		delete[] fr[i];
		delete[] isb[i];
	}
	delete[] fludif;
	delete[] rsumr;
	delete[] utilde;
	delete[] htilde;
	delete[] absvt;
	delete[] uvdif;
	delete[] ssc;
	delete[] vsc;
	delete[] a;
	delete[] ac1;
	delete[] ac2;
	delete[] w;
	delete[] eiglam;
	delete[] sgn;
	delete[] fc;
	delete[] fl;
	delete[] fr;
	delete[] ptest;
	delete[] isb;
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

