import numpy as np
from mesh import LinearMesh2D
from pml import PML
from rickerPulse import RickerPulse
from externalForce import ExternalForce
from ElementFrame2D import ElementFrame
from utilities import nodalPos
from timeSolver import *
from plot import *

class Problem:
    def __init__(self,
                 el: int, ed: int, length: float, depth: float,
                 I: float, freq: float, T: float, dt: float,
                 sourcesPos: list,
                 receiversPos: list,
                 materialModel: object,
                 ABC: tuple = None
                 ):

        # GENERATE MESH
        self.mesh: object = LinearMesh2D([el,ed,length,depth])

        # MAKE SOURCES AND RECEIVERS
        self.sources   = nodalPos(sourcesPos, self.mesh)
        self.receivers = nodalPos(receiversPos, self.mesh)

        # APPLY ABSORBING CONDITIONS
        if ABC is None:
            self.absLayer: object = PML(self.mesh, thickness=0, atten=1)
        else:
            self.absLayer: object = PML(self.mesh, thickness=ABC[0], atten=ABC[1])

        # SET EXTERNAL FORCES
        self.pulse: object = RickerPulse([I,freq,T,dt,1.0])
        self.force: object = ExternalForce(self.sources, self.mesh.nNodes, self.pulse.pulse)

        # GET FINITE ELEMENT STIFFNESS FRAME
        self.frame: object = ElementFrame(self.mesh, sparse_mode=True)

        # SET MATTERIAL MODEL
        self.materialModel = materialModel

        # SET MAIN INSTANCE ATTRIBUTES
        self.nn = self.mesh.nNodes
        self.ne = self.mesh.nElements
        self.dt = self.pulse.deltaTime
        self.st = self.pulse.steps
        self.T  = self.pulse.timeOfObservation
        self.ns = self.sources.shape[0]
        self.nr = self.receivers.shape[0]


    def solve(self,diag_scale=None,render=None,saveReceivers=None):

        M     = np.zeros([self.nn, 1], dtype=np.float32)
        mCons = np.zeros([self.nn, self.nn], dtype=np.float32)
        phiE  = np.zeros([4, 1], dtype=np.float32)

        # SWEEPING THROUGH ELEMENTS:
        for n in range(0, self.ne):
            mu = self.materialModel.get_eMu(n,self.mesh)
            for i in range(0, 4):
                for j in range(0, 4):
                    mCons[self.mesh.Connect[n, i] - 1, self.mesh.Connect[n, j] - 1] += mu * self.frame.frameMass[i, j]

        # LUMPING MASS (ROW-SUM):
        if diag_scale is None:
            M = mCons.sum(axis=1).reshape((self.nn, 1))
        else:
            # LUMPING MASS (DIAGONAL SCALING):
            mtot = mCons.sum()
            c = mtot / np.trace(mCons)
            for i in range(0, self.nn):
                M[i, 0] = c * mCons[i, i]

        u = np.zeros([self.nn, self.st, self.ns], dtype=np.float32)
        for shot in range(0, self.ns):
            u[:, :, shot] = solverEXP1shotPML_CCompiled(self.frame.stiff, M,
                                                        self.force.force,
                                                        self.dt,
                                                        self.sources[shot], shot,
                                                        self.absLayer).base

        if saveReceivers is not None:
            exp = np.float32(np.zeros([self.nr, self.st, self.ns]))
            for shot in range(0, self.ns):
                for receiver in range(0, self.nr):
                    exp[receiver, :, shot] = u[self.receivers[receiver] - 1, :, shot]
            # TODO: DEVE SALVAR RECEIVER AQUI

        if render is not None:
            render_propagating(self.mesh, u[:, :, 0], size=10)


    def get_grad(self):
        pass




