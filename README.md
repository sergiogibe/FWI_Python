## An FWI implementation in Python..

The Full Waveform Inversion method was first introduced by Lailly and Tarantola. It consists of a comparative analysis between the response of a simulated model and that of experimental data collected. This is characterized as a highly non-linear PDE- constrained optimization problem, which in many cases presents a particularly difficult solution to be found. Recent works show a recurrent difficulty in identifying more complex profiles, justifying the constant search for improvement of the inversion procedure. Besides this, the solution to the transient propagation problem makes the method very costly in terms of processing time and memory usage, therefore the solution of this problem is the target of several studies on HPC (High-Performance Computing). Regarding the application of the method, the FWI acoustic model has been a commonly used tool in the oil and gas industry to estimate velocity models.

## About this code:

The purpose of this code is to help students learn the FWI method. The Python language was chosen precisely for its clean syntax. To make it also possible to simulate larger problems a solution using the Cython module was implemented. Using the Cython library allows the solvers to be compiled in C. Also, in this way, you can utilize the advantages of multi-threading with the GIL release. This way it is possible to keep the syntax close to Python and get a very solid performance. The implementation is based on the finite element method and performs the update of the distribution function by using the reaction-diffusion method. 

## Required libraries:
- Stable on python 3.8.10 64-bit
- NumPy
- SciPy
- Matplotlib (Linux: Sometimes you will have to install GUI backends -  ```pip install pyqt5```)
- Cython (Make sure that you have Python 3.8.10 64-bit correctly installed.)
- Maybe you should install C/C++ compilers (Linux: ```sudo apt install build-essential```) (Windows: try installing ```Mingw-w64```, or intel compilers)

## First use. How to run?

To run this code properly you need to open and run: ```EXAMPLE_1.py```. 

Then, you can use the modules to make your own FWI implementation. You also need to remember to compile the solvers as it is described bellow.

## Altering Csolver code:
If you want to change anything in the solver go to ```Csolver.pyx```. After that you need to erase the last build by using the command: ```python setup.py clean --all``` or ```python3 setup.py clean --all``` in terminal inside the ```src``` folder. To compile your new code just type ```python setup.py build_ext --inplace``` or ```python3 setup.py build_ext --inplace```.

## General guidelines:

**Modules:**

- ```from source import Source``` and ```from receiver import Receiver``` - These modules creates the sources and the receivers.
- ```from rickerPulse import RickerPulse``` - As a default, this code uses the Ricker's pulse.
- ```from externalForce import ExternalForce``` - This module construct the external force based on the pulse used.
- ```from materialModel import MatmodelLS, MatmodelFWI, MatmodelSIMP``` - This module contains all the different material models classes.
- ```from ElementFrame2D import ElementFrame``` - This one initialize all the frames for the global matrices (M, K).
- ```from timeSolver import *``` - This module contains all the acoustic solvers.
- ```from plot import *``` - This modules calls some specific plot functions to help visualize the evolution of the problem. It is based on the Matplotlib library.
- ```from optimizationControl import backTracking_LineS``` - This funtion contains a modified Armijo line search method.

**These are some parameters you will need to set the problem:**

- lenght - Total lenght of the domain [km].
- depth  - Total depth of the domain [km].
- el - Number of elements lenght.
- ed - Number of elements depth.
- I - Pulse intensity.
- freq - Pulse frequency [hz].
- T - Time of observation [s].
- dt - Newmark time delta [s].
- delta - Reaction-diffusion pseudo-time.
- niter - Total number of problem iterations.
- diff - Reaction-diffusion coefficient.
- velocities - Medium velocities list (slowest to fastest) [km/s].
- normf - Sensitivity normalizing factor.
- regf  - Sensitivity regularization factor.

**You can choose between these material models:**

- "FWI"  - for traditional not penalized FWI distribution function
- "LS" and "MMLS"  - for Level-Set implicit surface. Recommended to identify sharp interfaces
- "SIMP" - for Solid Isotropic Material with Penalization



## Tasks to be accomplished:

- [x] Improve forward solver speed.
- [x] Implement perfectly matched layer (absorbing boundary).
- [ ] Finish the implementation of other material models.
- [ ] Implement BFGS.
- [ ] Improve mounting and updating speed (especially level set curve fitting).
- [ ] Implement continuation process for changing material model. 
- [ ] Implement sensitivity material filter.
- [ ] Implement interface to configure sources and receivers.
- [ ] Implement this code for frequency domain.
- [ ] Implement elastic-wave model.


## Contact:

sergiovitor@poli.ufrj.br

or

sergio.vitor@posgrad.ufsc.br











