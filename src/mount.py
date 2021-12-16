import numpy as np

class MountProblem:

    def __init__(self, meshObj, matModelObj, frameObj, dataGen, diag_scale):

        self.mesh     = meshObj
        self.matModel = matModelObj
        self.frame    = frameObj
        self.dataGen  = dataGen

        self.mass = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)
        m = np.zeros([self.mesh.nNodes, self.mesh.nNodes], dtype=np.float32)

        #Sweeping elements:
        for n in range(0, self.mesh.nElements):

            mu = self.matModel.apply_property(n,dataGen=self.dataGen)

            for i in range(0, 4):
                for j in range(0, 4):
                    m[self.mesh.Connect[n, i] - 1, self.mesh.Connect[n, j] - 1] += mu * self.frame.frameMass[i, j]

        #Lumping mass (row-sum):
        if not diag_scale:
            self.mass = m.sum(axis=1).reshape((self.mesh.nNodes, 1))

        #Lumping mass (diagonal scaling):
        if diag_scale:
            mtot = m.sum()
            c = mtot/np.trace(m)
            for i in range(0, self.mesh.nNodes):
                self.mass[i,0] = c*m[i,i]