import numpy as np
import os
from .mesh import LinearMesh2D
from .pml import PML
from .rickerPulse import RickerPulse
from .externalForce import ExternalForce
from .ElementFrame2D import ElementFrame
from .utilities import nodalPos
from .timeSolver import *
from .plot import *




class Problem:
    def __init__(self,
                 el: int, ed: int, length: float, depth: float,
                 I: float, freq: float, T: float, dt: float,
                 sourcesPos: list,
                 receiversPos: list,
                 materialModel: object,
                 ABC: tuple = None,
                 saveResponse: bool = False,  # Only used for generating data (exp problem).
                 ) -> None:
        """  This class is responsible for:
        - Genarating the linear square elements mesh
        - Creating the sources, receivers, and absorbing layers
        - Generating the pulse (and external force)
        - Calculating the FEM matrices frame and creating the problem
        for the given model (e.g. LevelSet class object)
        - Solving forward problem (with a switch in case of experimental 
        problem, which saves the data in data_dir directory), and adjoint problem.


        * The solvers first calculate the mass matrix, which is the
        one that is not constant throughout the iterations. 
        * The solvers are implemented in the Csolver.pyx file, using the 
        Cython library. For any changes being effective, you will need to 
        recompile the program as shown in README.md.
        """

        # GENERATE MESH
        self.mesh: object = LinearMesh2D([el,ed,length,depth])

        # MAKE SOURCES AND RECEIVERS
        self._sources   = nodalPos(sourcesPos, self.mesh)
        self._receivers = nodalPos(receiversPos, self.mesh)

        # APPLY ABSORBING CONDITIONS
        if ABC is None:
            self._absLayer: object = PML(self.mesh, thickness=0, atten=1)
        else:
            self._absLayer: object = PML(self.mesh, thickness=ABC[0], atten=ABC[1])

        # SET EXTERNAL FORCES
        self._pulse: object = RickerPulse([I,freq,T,dt,1.0])
        self._force: object = ExternalForce(self._sources, self.mesh.nNodes, self._pulse.pulse)

        # GET FINITE ELEMENT STIFFNESS FRAME
        self.frame: object = ElementFrame(self.mesh, sparse_mode=True)

        # SET MATTERIAL MODEL
        self.materialModel: object = materialModel

        # SET UP FOR EXP PROBLEM
        self.save: bool = saveResponse

        # SET MAIN INSTANCE ATTRIBUTES
        self._nn: int   = self.mesh.nNodes
        self._ne: int   = self.mesh.nElements
        self._dt: float = self._pulse.deltaTime
        self._T:  float = self._pulse.timeOfObservation
        self._st: int   = self._pulse.steps
        self._ns: int   = self._sources.shape[0]
        self._nr: int   = self._receivers.shape[0]



    def solve(self,
              diag_scale: bool = True,
              render: bool = None,
              exp = None
              ) -> None:
        """Solves forward problem (with a switch in case of experimental 
        problem, which saves the data in data_dir directory)
        and adjoint problem."""

        if exp is not None:
            self.exp = np.float32(exp)

        M     = np.zeros([self._nn, 1], dtype=np.float32)
        mCons = np.zeros([self._nn, self._nn], dtype=np.float32)
        phiE  = np.zeros([4, 1], dtype=np.float32)

        # SWEEPING THROUGH ELEMENTS:
        for n in range(0, self._ne):
            mu = self.materialModel.eMu(n,self.mesh)
            for i in range(0, 4):
                for j in range(0, 4):
                    mCons[self.mesh.Connect[n, i] - 1, self.mesh.Connect[n, j] - 1] += mu * self.frame.frameMass[i, j]


        # LUMPING MASS (DIAGONAL SCALING):
        if diag_scale:
            mtot = mCons.sum()
            c = mtot / np.trace(mCons)
            for i in range(0, self._nn):
                M[i, 0] = c * mCons[i, i]
        # LUMPING MASS (ROW-SUM):
        else:
            M = mCons.sum(axis=1).reshape((self._nn, 1))


        # EXPERIMENTAL
        if self.save:

            u = np.zeros([self._nn, self._st, self._ns], dtype=np.float32)
            self.exp = np.float32(np.zeros([self._nr, self._st, self._ns]))
            for shot in range(0, self._ns):
                print(f"SOLVING EXP       - SHOT {shot+1} ")
                u[:, :, shot] = solverEXP1shotPML_CCompiled(self.frame.stiff, M,
                                                            self._force.force,
                                                            self._dt,
                                                            self._sources[shot], shot,
                                                            self._absLayer).base

                for receiver in range(0, self._nr):
                    self.exp[receiver, :, shot] = u[self._receivers[receiver] - 1, :, shot]
            with open(f'{os.getcwd()}/data_dir/exp_data.npy', 'wb') as f:
                np.save(f, self.exp)

            if render is not None:
                render_propagating(self.mesh, u[:, :, 0], size=10)

        # FORWARD + OBJECTIVE
        else:

            self.obj = 0
            self.sens = np.zeros([self._nn, 1], dtype=np.float32)
            for shot in range(0, self._ns):
                print(f"SOLVING FORWARD   - SHOT {shot + 1} ")
                v, obj, misfit = solverF1shot_CCompiled(self.frame.stiff, M,
                                                        self._force.force,
                                                        self._dt,
                                                        self._sources[shot], shot,
                                                        self._receivers,
                                                        self.exp
                                                        )

                print(f"SOLVING ADJOINT   - SHOT {shot + 1} ")
                sens = solverS_CCompiled(self.frame.stiff, M,
                                         misfit,
                                         self._dt,
                                         self._receivers,
                                         v
                                         )

                self.obj  += obj
                self.sens += sens





