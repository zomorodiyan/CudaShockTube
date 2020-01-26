
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "ShockTube.cuh"


int main() {
	ShockTube st;

	// TEST
	st.HostTest01(); // tests Memory Allocation And sod shock tube problem Initialization on Host | nbrOfGrids == 10
	st.HostTest02(); // tests reflection Boundary Condition on Host | nbrOfGrids == 10
	st.HostTest03(); // tests one step of laxWendroff on Host | nbrOfGrids == 10 
	st.HostTest04(); // tests one step of Roe and Pike on Host | nbrOfGrids == 10 
	st.DeviceTest01(); // tests Memory Allocation And sod shock tube problem Initialization on Device | nbrOfGrids == 10
	st.DeviceTest02(); // tests reflection Boundary Condition on Device | nbrOfGrids == 10
	st.DeviceTest03(); // tests one step of laxWendroff on Device | nbrOfGrids == 10 
	st.DeviceTest04(); // tests one step of Roe and Pike on Device | nbrOfGrids == 10 

	// SOLUTION
	st.LaxHost(); // solves the Sod shock tube problem using laxWendroff steps for t == 0.2 on Host
	st.RoeHost(); // solves the Sod shock tube problem using Roe and Pike steps and SuperBee flux limiter for t == 0.2 on Host
	st.LaxDevice(); // solves the Sod shock tube problem using laxWendroff steps for t == 0.2 on Device 
	st.RoeDevice(); // solves the Sod shock tube problem using Roe and Pike steps and SuperBee flux limiter for t == 0.2 on Device

	return 0;
}