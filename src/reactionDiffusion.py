import scipy.sparse
import scipy.sparse.linalg

class RD:
    def __init__(self,
                 tau: float, delta: float,
                 frame: object,
                 dirichlet: bool = True
                 ):

        # SET UP CONSISTENT MATRICES STIFF AND DAMP
        self.dirichlet         = dirichlet
        self.dstiff, self.damp = frame.getConsistent(tau=tau,dirichlet=dirichlet)
        self.stiff, _          = frame.getConsistent(dirichlet=False)

        # SET MAIN INSTANCE ATTRIBUTES
        self.frame = frame
        self.tau   = tau
        self.delta = delta
        self.nn    = self.damp.shape[0]


    def solve(self, sens, control):

        # APPLY DIRICHLET RESTRICTIONS TO SENS
        if self.dirichlet:
            for node in self.frame.nodes_DirichletL:
                sens[node-1,0] = 0.0

        # SOLVE REACTION-DIFFUSION EQUATION
        lsK     = self.dstiff + (1 / self.delta) * self.damp
        lsF     = self.damp.dot((1 / self.delta) * control + sens.reshape((self.nn,1)))
        control = scipy.sparse.linalg.spsolve(lsK, lsF).reshape((self.nn, 1))

        return control