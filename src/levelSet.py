import numpy as np
from curveFitting import CurveFitting

class LevelSet:
    def __init__(self,
                 el: int, ed: int,
                 velocities: list,
                 diff: float,
                 delta: float
                 ):

        # SET MAIN INSTANCE ATTRIBUTES
        self.nn         = (el+1)*(ed+1)
        self.diff       = diff
        self.delta      = delta
        self.control    = np.ones([self.nn, 1], dtype=np.float32)*(-0.01)
        self.velocities = velocities
        if len(velocities) > 2:
            exit(print('This LevelSet Class accounts for 2 materials only. For more try MMLevelSet Class.'))
        self.mus        = [1 / pow(self.velocities[i], 2) for i in range(0, 2)]

    def get_eMu(self,
                eNumber: int,   # eNumber starts at 0
                mesh: object
                ):

        # FIND PROPERTY FOR ELEMENT (LEVEL SET - FIT CURVE)
        eControl = np.zeros([4, 1], dtype=np.float32)

        for node in range(0, 4):
            eControl[node, 0] = self.control[mesh.Connect[eNumber, node] - 1, 0]

        fit = CurveFitting(eControl)
        fit.fitLevelSet(order=2, gridSize=3)
        eMu = fit.distribute(self.mus[1], self.mus[0])

        return eMu