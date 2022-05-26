import numpy as np

class LinearMesh2D:

    def __init__(self, envConfig):

        self.nElementsL      = envConfig[0]
        self.nElementsD      = envConfig[1]
        self.length          = envConfig[2]
        self.depth           = envConfig[3]

        self.nNodes          = (envConfig[0] +1) * (envConfig[1] +1)
        self.nElements       = (envConfig[0])   * (envConfig[1])

        self.mCoord          = np.zeros([self.nNodes ,2],dtype=np.float32)
        self.Connect         = np.zeros([self.nElements ,4], dtype=int)

        auxD = self.depth /self.nElementsD
        auxL = self.length /self.nElementsL

        for i in range(0 ,self.nElementsL +1):
            self.mCoord[i ,0] = auxL *i
            for j in range(0 ,self.nElementsD +1):
                self.mCoord[ i +(self.nElementsL +1 ) *j ,0] = self.mCoord[i ,0]

        for n in range(0 ,self.nElementsD +1):
            for m in range(0 ,self.nElementsL +1):
                self.mCoord[ m +(self.nElementsL +1 ) *n ,1] = auxD *n

        for n in range(0 ,self.nElementsD):
            for i in range(0 ,self.nElementsL):
                self.Connect[i + self.nElementsL * n, 0] = 1 + i + self.nElementsL * n + n
                self.Connect[i + self.nElementsL * n, 1] = 2 + i + self.nElementsL * n + n
                self.Connect[i + self.nElementsL * n, 2] = self.Connect[i + self.nElementsL * n, 1] + \
                            (self.nElementsL + 1)
                self.Connect[i + self.nElementsL * n, 3] = self.Connect[i + self.nElementsL * n, 0] + (
                            self.nElementsL + 1)
