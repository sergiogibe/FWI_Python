import numpy as np
from .curveFitting import CurveFitting
from .plot import plot_contour
import scipy.sparse
import scipy.sparse.linalg
from .misc import COLORS

class MMLevelSet:
    def __init__(self,
                 el: int, ed: int,
                 length: float, depth: float,
                 velocities: list
                 ):
        # SET MAIN INSTANCE ATTRIBUTES
        self.el, self.ed = el, ed
        self.length = length
        self.depth = depth
        self.nn = (el + 1) * (ed + 1)
        self.velocities = velocities    # Higher velocities come before on the list.
        self.mus = [1 / pow(velocities[i], 2) for i in range(0, 2)]
        self.nm = len(velocities)

        self.control = np.ones([self.nn, 1], dtype=np.float32)*0.00001
        self.levels = [(0.0 + i*(1/self.nm)) for i in range(self.nm)]
        self.levels.append(1.0)

    def eMu(self,
            eNumber: int,  # eNumber starts at 0
            mesh: object
            ):

        # FIND PROPERTY FOR ELEMENT (LEVEL SET - FIT CURVE)
        avg = 0
        for node in range(4):
            avg += 0.25 * self.control[mesh.Connect[eNumber, node] - 1, 0]

        # EXAMPLE WITH 4 MATERIALS:
        # 1 material - control function interval = [0.00,0.25]
        # 2 material - control function interval = [0.25,0.50]
        # 3 material - control function interval = [0.50,0.75]
        # 4 material - control function interval = [0.75,1.00]
        # self.levels = [0.00, 0.25, 0.50, 0.75, 1.00]

        if avg < 0.0: #TODO: AQUI NAO E O LUGAR CORRETO DE FAZER ISSO (PLACEHOLDER)
            avg == 0.0
        if avg > 1.0:
            avg == 1.0

        eMu = 1
        for i in range(1,len(self.levels)-1):
            if avg > self.levels[i-1] and avg <= self.levels[i]:
                eMu = self.mus[i-1]

        return eMu


    def plot(self, nametag: str,
             save: bool=True,
             plotSR: list=None):
        vp = np.zeros((self.el+1,self.ed+1))
        for j in range(self.ed+1):
            for i in range(self.el+1):
                vp[j,i] = self.control[i+j*(self.el+1)]
        plot_contour(2, str(nametag), vp, (-1,1), levels=self.levels, colors=COLORS,
                     fill=True, extent = (self.length,self.depth), plotSR=plotSR, save=save)


    def update(self,
               sens: np.array,
               kappa: float, c: float,
               reacDiff: object,
               regSens: bool = True,
               normSens: bool = True,
               savePlot: bool = True,
               nametag: str = "NONAME",
               plotSR: list = None
               ):
        print("UPDATING MODEL")
        # APPLY MATERIAL MODEL DERIVATIVE WITH RESPECT TO CONTROL FUNCTION TODO VERIFICAR ISSO
        H = np.zeros((self.nn,1), dtype=np.float32)
        for i in range(2,len(self.levels)-1):
            H[np.where(np.logical_and(self.control <= self.levels[i],
                                      self.control > self.levels[i-1]))] += self.mus[i-2] - self.mus[i-1]
        sens = np.multiply(H,sens)

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