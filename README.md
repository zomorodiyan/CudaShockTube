# SodShockTube
### Simulation of Sod shock tube problem, using C++,on CPU and NVIDIA CUDA, on GPU
![SodShockTubeOnCudaNvidia](http://zomorod.me/wp-content/uploads/2019/10/Sod.jpg)<br/>
The Sod shock tube problem, named after Gary A. Sod, is a common test for the accuracy of computational fluid codes, like Riemann solvers, and was heavily investigated by Sod in 1978. The test consists of a one-dimensional Riemann problem which its time evolution can be described by solving the Euler equations, which leads to three characteristics, describing the propagation speed of the various regions of the system. Namely the expansion wave, the contact discontinuity and the shock discontinuity. If this is solved numerically, one can test against the analytical solution, and get information how well a code captures and resolves shocks and contact discontinuities and reproduce the correct density profile of the expansion wave.  
In this project we numerically solve the Sod shock tube problem using CUDA and compare the results with the anaylitical solution. Moreover, we compare the run-time and memory cost of solution on cpu in C++ and solution on gpu in CUDA C++.
Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
![](http://zomorod.me/wp-content/uploads/2020/02/Untitled1-Recovered.gif)  |  ![](http://zomorod.me/wp-content/uploads/2020/02/Untitled1-Recovered.gif)
