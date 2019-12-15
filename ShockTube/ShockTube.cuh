#pragma once

class ShockTube
{
public:
	ShockTube() { ; }
	~ShockTube(){ ; }
	void logcu();
	void logcpp();
	// Allocate space for host copies of the variables
	void allocHostMemory();

	// Assigns Sod's shock tube problem initial conditions
	void initialize();

	// Calculate and return average values of u, v, p, e
	double* getAverages();

	// set number of grid points (default = 200)
	void setNbrOfGrids(int nbr) { nbrOfGrids = nbr; }

private:
	// data members
	int nbrOfGrids = 200;	// number of grid points (default = 200)
	double h, length, gama, cfl, nu, tau, step, uAverage, roAverage, eAverage, pAverage;
	double *u1, *u2, *u3, *f1, *f2, *f3, *vol;
	int size = nbrOfGrids * sizeof(int);

	// Calculate cMax
	double cMax();
};
