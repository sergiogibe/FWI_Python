import numpy as np
import scipy.linalg

class CurveFitting:

    def __init__(self,eLS):

        self.ls = eLS
        self.lsRatio    = 1

    def fitLevelSet(self,order,gridSize):

        x = [0,1,1,0]
        y = [0,0,1,1]
        z = [self.ls[0, 0], self.ls[1, 0], self.ls[2, 0], self.ls[3, 0]]
        z.sort()
        data = np.c_[x, y, z]

        mn = np.min(data, axis=0)
        mx = np.max(data, axis=0)
        X, Y = np.meshgrid(np.linspace(mn[0], mx[0], gridSize), np.linspace(mn[1], mx[1], gridSize))
        XX = X.flatten()
        YY = Y.flatten()

        if order == 1:
            A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
            C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
            Z = C[0] * X + C[1] * Y + C[2]
        if order == 2:
            A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
            C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
            Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)
        else:
            Z = np.zeros([gridSize,gridSize])

        counter = 0
        for i in range(0,Z.shape[0]):
            for j in range(0,Z.shape[1]):
                if Z[i,j] >= 0:
                    counter += 1

        self.lsRatio = counter/(np.size(Z))

    def distribute(self,parameter1,parameter2):

        return (parameter1*self.lsRatio) + (parameter2*(1-self.lsRatio))