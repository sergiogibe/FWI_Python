import numpy as np
import scipy.sparse
import scipy.sparse.linalg

class ElementFrame:

    def __init__(self, meshObj, sparse_mode):

        self.mesh = meshObj
        self.sparse_mode = sparse_mode
        self.frameCoord = np.zeros([4, 2],dtype=np.float32)
        self.frameStiff = np.zeros([4, 4],dtype=np.float32)
        self.frameMass = np.zeros([4, 4],dtype=np.float32)
        self.stiff = np.zeros([self.mesh.nNodes, self.mesh.nNodes],dtype=np.float32)

        self.getFrame()

    def getCoordinates(self):

        for j in range(0, 4):
            self.frameCoord[j, 0] = self.mesh.mCoord[(self.mesh.Connect[0, j]) - 1, 0]
            self.frameCoord[j, 1] = self.mesh.mCoord[(self.mesh.Connect[0, j]) - 1, 1]

    def getStiffness(self):

        pointsAW = gaussPointsWeights()
        for i in range(0, 4):
            r = pointsAW[i, 0]
            s = pointsAW[i, 1]
            weight = pointsAW[i, 2]
            B,det,_ = shapeFunctions(r, s, self.frameCoord)
            self.frameStiff += (np.matmul(np.transpose(B), B)) * det * weight

        #Generate global K:
        for n in range(0, self.mesh.nElements):
            for i in range(0, 4):
                for j in range(0, 4):
                    self.stiff[self.mesh.Connect[n, i] - 1, self.mesh.Connect[n, j] - 1] += self.frameStiff[i, j]

        #Sparse stiffness:
        if self.sparse_mode:
            offsets = [-self.mesh.nElementsL - 2, -self.mesh.nElementsL - 1, -self.mesh.nElementsL,
                        -1, 0, 1,
                        self.mesh.nElementsL, self.mesh.nElementsL + 1, self.mesh.nElementsL + 2]
            diagonalsStiff = []

            for i in range(0, len(offsets)):
                diagonalsStiff.append(self.stiff.diagonal(offsets[i]))

            self.stiff = scipy.sparse.diags(diagonalsStiff, offsets, format='csr',dtype=np.float32)

    def getMass(self):

        pointsAW = gaussPointsWeights()
        for i in range(0, 4):
            r = pointsAW[i, 0]
            s = pointsAW[i, 1]
            weight = pointsAW[i, 2]
            _,det,N = shapeFunctions(r, s, self.frameCoord)
            self.frameMass += (np.matmul(np.transpose(N), N)) * det * weight

    def getFrame(self):

        self.getCoordinates()
        self.getMass()
        self.getStiffness()

    def getConsistent(self,tau):

        #Helpful info: tau = diffusion coefficient
        #              (in case of tau = 0 -> get matrices only for sensitivity regularization)

        stiff = np.zeros([self.mesh.nNodes, self.mesh.nNodes],dtype=np.float32)
        damp  = np.zeros([self.mesh.nNodes, self.mesh.nNodes],dtype=np.float32)

        for n in range(0, self.mesh.nElements):
            element = ElementConsistent(n + 1, self.mesh)
            for i in range(0, 4):
                for j in range(0, 4):
                    stiff[self.mesh.Connect[n, i] - 1, self.mesh.Connect[n, j] - 1] += tau*element.eStiff[i, j]
                    damp[self.mesh.Connect[n, i] - 1, self.mesh.Connect[n, j] - 1] += element.eDamp[i, j]

        offsets = [-self.mesh.nElementsL - 2, -self.mesh.nElementsL - 1, -self.mesh.nElementsL,
                   -1, 0, 1,
                   self.mesh.nElementsL, self.mesh.nElementsL + 1, self.mesh.nElementsL + 2]

        diagonalsStiff = []
        diagonalsDamp = []

        for i in range(0, len(offsets)):
            diagonalsStiff.append(stiff.diagonal(offsets[i]))
            diagonalsDamp.append(damp.diagonal(offsets[i]))

        stiff = scipy.sparse.diags(diagonalsStiff, offsets, format='csc')
        damp = scipy.sparse.diags(diagonalsDamp, offsets, format='csc')

        return stiff, damp


class ElementConsistent:

    def __init__(self, wE, meshObj):

        self.wE   = wE
        self.mesh = meshObj

        self.eCoord = np.zeros([4, 2],dtype=np.float32)
        self.eStiff = np.zeros([4, 4],dtype=np.float32)
        self.eDamp = np.zeros([4, 4],dtype=np.float32)

        for j in range(0, 4):
            self.eCoord[j, 0] = self.mesh.mCoord[(self.mesh.Connect[self.wE - 1, j]) - 1, 0]
            self.eCoord[j, 1] = self.mesh.mCoord[(self.mesh.Connect[self.wE - 1, j]) - 1, 1]

        pointsAW = gaussPointsWeights()

        for i in range(0, 4):
            r = pointsAW[i, 0]
            s = pointsAW[i, 1]
            weight = pointsAW[i, 2]
            B, det, N = shapeFunctions(r, s, self.eCoord)
            self.eStiff += (np.matmul(np.transpose(B), B)) * det * weight
            self.eDamp += (np.matmul(np.transpose(N), N)) * det * weight



def shapeFunctions(r, s, X):

    N = np.zeros([1, 4],dtype=np.float32)
    N[0, 0] = 0.25 * (1 - r) * (1 - s)
    N[0, 1] = 0.25 * (1 + r) * (1 - s)
    N[0, 2] = 0.25 * (1 + r) * (1 + s)
    N[0, 3] = 0.25 * (1 - r) * (1 + s)

    psiGradRS = np.zeros([2, 4],dtype=np.float32)
    psiGradRS[0, 0] = -0.25 * (1 - s)
    psiGradRS[0, 1] = 0.25 * (1 - s)
    psiGradRS[0, 2] = 0.25 * (1 + s)
    psiGradRS[0, 3] = -0.25 * (1 + s)
    psiGradRS[1, 0] = -0.25 * (1 - r)
    psiGradRS[1, 1] = -0.25 * (1 + r)
    psiGradRS[1, 2] = 0.25 * (1 + r)
    psiGradRS[1, 3] = 0.25 * (1 - r)

    J = np.matmul(psiGradRS, X)
    det = np.linalg.det(J)
    B = np.matmul(np.linalg.inv(J), psiGradRS)

    return B, det, N


def gaussPointsWeights():

    pointsAndWeights = np.zeros([4, 3],dtype=np.float32)

    gaussPointValue = 0.57735026905
    weight = 1.0

    pointsAndWeights[0, 0] = gaussPointValue
    pointsAndWeights[0, 1] = gaussPointValue
    pointsAndWeights[0, 2] = weight

    pointsAndWeights[1, 0] = gaussPointValue * (-1)
    pointsAndWeights[1, 1] = gaussPointValue
    pointsAndWeights[1, 2] = weight

    pointsAndWeights[2, 0] = gaussPointValue
    pointsAndWeights[2, 1] = gaussPointValue * (-1)
    pointsAndWeights[2, 2] = weight

    pointsAndWeights[3, 0] = gaussPointValue * (-1)
    pointsAndWeights[3, 1] = gaussPointValue * (-1)
    pointsAndWeights[3, 2] = weight

    return pointsAndWeights