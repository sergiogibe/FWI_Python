import scipy.sparse
import scipy.sparse.linalg

class RD:
    def __init__(self,
                 tau: float, delta: float,
                 frame: object
                 ) -> None:
        """This class is responsible for creating the
        Reaction-Diffusion object. This object updates the control
        function inside its own update method 
        (e.g. LevelSet Class update method)

        *The variables dstiff, damp, and stiff are the consistent matrices
        used to backup the reaction-diffusion operation and the
        sensitivity regularization procedure.
        """

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


class RD2:
    def __init__(self,
                 tau: float, delta: float,
                 frame: object,
                 dirichlet: bool = True
                 ) -> None:
        """This class is responsible for creating the
        Reaction-Diffusion object. This object updates the control
        function inside its own update method 
        (e.g. LevelSet Class update method)

        *The variables dstiff, damp, and stiff are the consistent matrices
        used to backup the reaction-diffusion operation and the
        sensitivity regularization procedure.
        *It is also possible to apply dirichlet boundary conditions on 
        the edges of the domain.
        """

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