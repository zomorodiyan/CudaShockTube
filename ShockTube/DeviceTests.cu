
#include <iostream>
#include <string>
#include "ShockTube.cuh"

#define fail "\033[1;31m"
#define pass "\033[1;32m"
#define reset "\033[0m"


void DeviceTest01() {
	const std::string test = "DeviceMemoryAllocatedCorrectly";
	std::cout << __func__ << std::endl;
	ShockTube st;
	st.setNbrOfGrids(10);
	st.allocDeviceMemory();
	std::cout << pass << test << reset << std::endl;
}

