import scipy.sparse
import scipy.sparse.linalg

class RD:
    def __init__(self,
                 tau: float, delta: float,
                 frame: object
                 ):

        # SET UP CONSISTENT MATRICES STIFF AND DAMP
        self.dstiff, self.damp = frame.getConsistent(tau)
        self.stiff, _          = frame.getConsistent(1.0)

        # SET MAIN INSTANCE ATTRIBUTES
        self.tau   = tau
        self.delta = delta
        self.nn    = self.damp.shape[0]


    def solve(self, sens, control):
        lsK     = self.dstiff + (1 / self.delta) * self.damp
        lsF     = self.damp.dot((1 / self.delta) * control + sens.reshape((self.nn,1)))
        control = scipy.sparse.linalg.spsolve(lsK, lsF).reshape((self.nn, 1))

        return control