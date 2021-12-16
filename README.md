## A FWI implementation in python..

The Full Waveform Inversion method was first introduced by Lailly and Tarantola. It consists of a comparative analysis between the response of a simulated model and that of experimental data collected. This is characterized as a highly non-linear optimization problem, which in many cases presents a particularly difficult solution to be found. Recent works show a recurrent difficulty in identifying more complex profiles, justifying the constant search for improvement of the inversion procedure. Besides this, the solution to the transient propagation problem makes the method very costly in terms of processing time and memory usage, therefore the solution of this problem is the target of several studies on HPC (High-Performance Computing). Regarding the application of the method, the FWI acoustic model has been a commonly used tool in the oil and gas industry to estimate velocity models.

About the propagation model, there are two typical approaches. Some authors have presented an extension of FWI using the elastic wave model. This model can represent shear-type waves, while the simplified or acoustic model considers only longitudinal waves. The choice of the material model can also provide distinct results in the final inversion.

## About this code:

The purpose of this code is to help students learn the FWI method. The Python language was chosen precisely for its clean syntax. To make it also possible to simulate larger problems a solution using the Cython module was implemented. Using the Cython library allows the solvers to be compiled in C. Also, in this way, you can utilize the advantages of multi-threading with the GIL release. This way it is possible to keep the syntax close to Python and get a very solid performance. The implementation is based on the finite element method and performs the update of the distribution function by using the reaction-diffusion method. 

## Required libraries:
- NumPy
- SciPy
- Matplotlib (Linux: Sometimes you will have to install GUI backends -  ```pip install pyqt5```)
- Cython (Make sure that you have the latest version of Python correctly installed.)

## First use. How to run?

To run this code properly you need to open and run one of the two examples: ```example1``` or ```example2```. 

Then, you can use the modules to make your own FWI implementation. 

## General guidelines:

**Modules:**

- ```from mesh import LinearMesh2D``` - This modules creates the 2D-mesh necessary for all the simulation.
- ```from source import Source``` and ```from receiver import Receiver``` - These modules creates the sources and the receivers. To dispose them on the domain it is necessary to make some kind of arrangement.
- ```from rickerPulse import RickerPulse``` - As a default, this code used the Ricker's pulse.
- ```from externalForce import ExternalForce``` - This module construct the external force based on the pulse used.
- ```from materialModel import MatmodelLS, MatmodelFWI, MatmodelSIMP``` - This module contains all the different material models classes.
- ```from ElementFrame2D import ElementFrame``` - This one initialize all the frames for the global matrices (M, K).
- ```from mount import MountProblem``` - This module mounts the problem considering the distribution of material properties (the mass is lumped).
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
- dispF - Ricker pulse's displacement.
- delta - Reaction-diffusion pseudo-time.
- niter - Total number of problem iterations.
- diff - Reaction-diffusion coefficient.
- upper_vel - Highest medium velocity [km/s].
- lower_vel - Lowest medium velocity [km/s].
- normf - Sensitivity normalizing factor.
- regf  - Sensitivity regularization factor.

**You can choose between these material models:**

- "FWI"  - for traditional not penalized FWI distribution function
- "LS"   - for Level-Set implicit surface. Recommended to identify sharp interfaces
- "SIMP" - for Solid Isotropic Material with Penalization

## Altering Csolver code:
If you want to change anything in the solver go to ```Csolver.pyx```. After that you need to erase the last build by using the command: ```python setup.py clean --all``` in terminal inside the ```src``` folder. To compile your new code just type ```python setup.py build_ext --inplace```.

## Tasks to be accomplished:

- [x] Improve forward solver speed.
- [ ] Implement perfectly matched layer (absorbing boundary).
- [ ] Finish the implementation of MMLS.
- [ ] Implement BFGS.
- [ ] Improve mounting and updating speed (especially level set curve fitting).
- [ ] Implement continuation process for changing material model. 
- [ ] Implement sensitivity material filter.
- [ ] Implement interface to configure sources and receivers.
- [ ] Consolidate other examples and setups.
- [ ] Implement this code for frequency domain.
- [ ] Implement a topological derivative approach.
- [ ] Implement elastic-wave model.

## Repository organization:

There are some examples in ```evolution_example``` of each material model running for 10 iteration with no step control.

## Contact:

sergiovitor@poli.ufrj.br











