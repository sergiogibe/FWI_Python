import numpy as np
from .curveFitting import CurveFitting
from .plot import plot_contour
from .misc import COLORS
import scipy.sparse
import scipy.sparse.linalg


class LevelSet:
    def __init__(self,
                 el: int, ed: int,
                 length: float, depth: float,
                 velocities: list
                 ) -> None:
        """  This class is responsible for:
        - Initializing the control function Level-Set for 2 materials
        - Calculating the element's slowness property (mu)
        - Calling the plot function (plot_contour from plot module)
        - Using the reacDiff object to update the level-set function
        """
        self.el, self.ed = el, ed
        self.length      = length
        self.depth       = depth
        self.nn          = (el+1)*(ed+1)
        self.control     = np.ones([self.nn, 1], dtype=np.float32)*(-0.01)
        self.velocities  = velocities
        if len(velocities) > 2:
            exit(print('This LevelSet Class accounts for 2 materials only. For more try MMLevelSet Class.'))
        self.mus         = [1 / pow(self.velocities[i], 2) for i in range(0, 2)]
        self.levels      = [-1.00, 0.00, 1.00]


    def eMu(self,
            eNumber: int,   
            mesh: object
            ) -> float:
        """Calculates the element's slowness property (mu)
           Useful info: eNumber starts at 0"""
        eControl = np.zeros([4, 1], dtype=np.float32)
        for node in range(0, 4):
            eControl[node, 0] = self.control[mesh.Connect[eNumber, node] - 1, 0]
        fit = CurveFitting(eControl)
        fit.fitLevelSet(order=2, gridSize=3)
        eMu = fit.distribute(self.mus[1], self.mus[0])
        return eMu


    def plot(self, nametag: str, save: bool=True, plotSR: list=None) -> None:
        """Calls the plot function (plot_contour from plot module)"""
        vp = np.zeros((self.el+1,self.ed+1))
        for j in range(self.ed+1):
            for i in range(self.el+1):
                vp[j,i] = self.control[i+j*101]
        plot_contour(2, str(nametag), vp, (-1,1), levels=self.levels, colors=COLORS,
                     fill=True, extent = (self.length,self.depth), plotSR=plotSR,
                     save=save)


    def update(self,
               sens: np.array,
               kappa: float, c: float,
               reacDiff: object,
               regSens: bool = True,
               normSens: bool = True,
               savePlot: bool = True,
               nametag: str = "NONAME",
               plotSR: list = None
               ) -> None:
        """Uses the reacDiff object to update the level-set function"""

        # APPLY MATERIAL MODEL DERIVATIVE WITH RESPECT TO CONTROL FUNCTION
        sens *= self.mus[1] - self.mus[0]

        # REGULARIZATION PROCEDURE
        if regSens:
            sfK = kappa * reacDiff.stiff + reacDiff.damp
            sfF = reacDiff.damp.dot(sens)
            sens = scipy.sparse.linalg.spsolve(sfK, sfF).reshape((self.nn, 1))

        # NORMALIZATION PROCEDURE
        if normSens:
            #d = abs(max(sens.min, sens.max, key=abs))
            d = 1
            for node in range(0,self.nn):
                sens[node,0] = c * (sens[node, 0] / d)

        # UPDATE CONTROL FUNCTION
        self.control = reacDiff.solve(sens,self.control)

        # PLOT
        self.plot(nametag, save=savePlot, plotSR=plotSR)