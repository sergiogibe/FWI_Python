## An FWI implementation in Python..

The Full Waveform Inversion method was first introduced by Lailly and Tarantola. It consists of a comparative analysis between the response of a simulated model and that of experimental data collected. This is characterized as a highly non-linear PDE- constrained optimization problem, which in many cases presents a particularly difficult solution to be found. Recent works show a recurrent difficulty in identifying more complex profiles, justifying the constant search for improvement of the inversion procedure. Besides this, the solution to the transient propagation problem makes the method very costly in terms of processing time and memory usage, therefore the solution of this problem is the target of several studies on HPC (High-Performance Computing). Regarding the application of the method, the FWI acoustic model has been a commonly used tool in the oil and gas industry to estimate velocity models.

## About this code:

The purpose of this code is to help students learn the FWI method. The Python language was chosen precisely for its clean syntax. To make it also possible to simulate larger problems a solution using the Cython module was implemented. Using the Cython library allows the solvers to be compiled in C. Also, in this way, you can utilize the advantages of multi-threading with the GIL release. This way it is possible to keep the syntax close to Python and get a very solid performance. The implementation is based on the finite element method and performs the update of the distribution function by using the reaction-diffusion method. 

## Required libraries:
- Stable on python 3.8.10 64-bit
- ```NumPy```
- ```SciPy```
- ```Matplotlib``` (Linux: Sometimes you will have to install GUI backends -  ```pip install pyqt5```)
- ```Cython``` (Make sure that you have the above python version correctly installed.)
- Maybe you should install C/C++ compilers (Linux: ```sudo apt install build-essential```) (Windows: try installing ```Mingw-w64```, or intel compilers)

## First use - Tutorial.py. How to run?

Test the demo code ```tutorial.py```. You need to run it from inside the ```FWI_Python/``` package (as your current working directory).

Then, if everything is working fine, you can use the module to make your own FWI implementation as shown in ```EXAMPLE_1``` and ```EXAMPLE_2``` in the ```examples/``` folder.

*Do not try to run the examples from inside the package directory because it will not work. This is the correct architecture:

```your_project_name/```
- ```your_implementation.py```
- ```FWI_Python/```
- ```plots/```
- ```data_dir/```

And you should import the package as follows: 

```import FWI_Python as fwi```

## Altering Csolver code:
If you want to change anything in the solver go to ```Csolver.pyx```. After that you need to erase the last build by using the command: ```python setup.py clean --all``` or ```python3 setup.py clean --all``` in terminal INSIDE the ```FWI_Python/src``` folder. To compile your new code just type ```python setup.py build_ext --inplace``` or ```python3 setup.py build_ext --inplace```.

## General guidelines:

**Classes:**

- ```fwi.LevelSet```:

    - Initializes the control function Level-Set for 2 materials.

    - Calculates the element's slowness property (```mu```).

    - Calls the plot function (```plot_contour``` from ```plot``` file).

    - Uses the ```RD``` object to update the level-set function.


- ```fwi.Problem```:

    - Genarates the linear square elements ```mesh``` object.

    - Creates the ```sources```, ```receivers```, and absorbing layers.

    - Generates the ```pulse``` (and external ```force```).
        
    - Calculates the FEM matrices frame and creating the problem
    for the given model (e.g. ```LevelSet``` class object).
    
    - Solves the forward problem (with a switch in case of experimental 
    problem, which saves the data in ```data_dir``` directory), and the adjoint problem.


        * The solvers first calculate the mass matrix, which is the
    one that is not constant throughout the iterations. 

        * The solvers are implemented in the ```Csolver.pyx``` file, using the 
    ```Cython``` library. For any changes being effective, you will need to 
    recompile the program as shown in ```README.md```.


- ```fwi.RD```:

    - Creates the Reaction-Diffusion object from ```RD``` class. This object updates the control function inside its own update method (e.g. ```LevelSet``` Class update method).

    - The variables ```dstiff```, ```damp```, and ```stiff``` are the consistent matrices used to backup the reaction-diffusion operation and the sensitivity regularization procedure.


**Parameters:**

- ```lenght``` - Total lenght of the domain [km].
- ```depth```  - Total depth of the domain [km].
- ```el``` - Number of elements lenght.
- ```ed``` - Number of elements depth.
- ```I``` - Pulse intensity.
- ```freq``` - Pulse frequency [hz].
- ```T``` - Time of observation [s].
- ```dt``` - Newmark time delta [s].
- ```delta``` - Reaction-diffusion pseudo-time.
- ```niter``` - Total number of problem iterations.
- ```diff``` - Reaction-diffusion coefficient.
- ```velocities``` - Medium velocities list (slowest to fastest) [km/s].
- ```normf``` - Sensitivity normalizing factor.
- ```regf```  - Sensitivity regularization factor.

## Ideas for the future:

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











