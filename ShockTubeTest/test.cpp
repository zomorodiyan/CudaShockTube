#include "pch.h"
//#include "../ShockTube/ShockTube.cuh"
#include "../ShockTube/ShockTube.cpp"


TEST(ShockTubeTest, HostMemoryAllocatedAndInitializedCorrectly) {
	ShockTube st;
	st.allocHostMemory();
	st.initialize();
	double* Averages = st.getAverages();
	double eps = 1e-14;
	ASSERT_TRUE((abs(Averages[0] - 0.566875) < eps)
		&& (abs(Averages[1] - 0) < eps)
		&& (abs(Averages[2] - 1.38625) < eps)
		&& (abs(Averages[3] - 0.5545) < eps));
}