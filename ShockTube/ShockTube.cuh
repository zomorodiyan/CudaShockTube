#pragma once
void HostTest01(); //void HostTest02(); //void HostTest03();
void DeviceTest01(); //void DeviceTest02(); //void DeviceTest03(); 

class ShockTube
{
public:
	ShockTube() { ; }
	~ShockTube(){ ; }
	// Allocate space for host copies of the variables
	void allocHostMemory();

	// Allocate space for host copies of the variables
	void allocDeviceMemory();

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
	double *d_u1, *d_u2, *d_u3, *d_f1, *d_f2, *d_f3, *d_vol;
	int size = nbrOfGrids * sizeof(int);

	// Calculate cMax
	double cMax();
};
