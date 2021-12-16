import numpy as np
from matplotlib import pyplot as plt
from curveFitting import CurveFitting
import scipy.sparse
import scipy.sparse.linalg


class Matmod:

    def __init__(self, meshObj, optConfig):

        self.mesh       = meshObj
        self.delta      = optConfig[0]
        self.niter      = optConfig[1]

        self.model      = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)
        self.modelHist  = np.zeros([self.mesh.nNodes, 0], dtype=np.float32)

    # SUPER CLASS METHODS
    def writeHist(self):

        self.modelHist = np.c_[self.modelHist, self.model]


    def update(self, sens, difA, difB):

        #Reaction-Diffusion Evolution
        lsK = difA + (1/self.delta)*difB
        lsF = difB.dot((1/self.delta)*self.model.reshape((self.mesh.nNodes,1)) + sens.reshape((self.mesh.nNodes,1)))

        self.model = scipy.sparse.linalg.spsolve(lsK,lsF).reshape((self.mesh.nNodes,1))




class MatmodelFWI(Matmod):

    # mu1 <= PHI <= mu2

    def __init__(self, meshObj, optConfig):

        super().__init__(meshObj, optConfig)
        self.realModel      = np.ones([self.mesh.nNodes, 1], dtype=np.float32) * (1/pow(self.mesh.propSpeed[1],2))
        self.delta *= 5 #NORMALLY FWI STEP NEEDS TO BE BIGGER

    def square_dist(self):

        side         = self.mesh.nElementsL//8
        height       = self.mesh.nElementsL//8
        firstElement = self.mesh.nElementsL//2 - side//2 + self.mesh.nElementsL*(self.mesh.nElementsL//2 - height//2)
        listOfElements = []

        for i in range(0,height):
            for j in range(0,side):
                listOfElements.append(firstElement + i*self.mesh.nElementsL + j)
        listOfElements.sort()

        for i in range(0, len(listOfElements)):
            for j in range(0, 4):
                self.realModel[self.mesh.Connect[listOfElements[i] - 1, j] - 1, 0] = 1/pow(self.mesh.propSpeed[0],2)

        self.plot_realModel()

    def homoGuess(self):

        self.model[:,0] = 1/pow(self.mesh.propSpeed[1],2)
        self.modelHist  = np.c_[self.modelHist,self.model]

        self.plot_model(ID=0)

    def read_RealModel(self):

        with open('../FWI_Python/data_dir/real_model.npy', 'rb') as f:
            self.realModel = np.float32(np.load(f))
        self.plot_realModel()

    def read_Guess(self):

        with open('../FWI_Python/data_dir/guess.npy', 'rb') as f:
            self.model = np.float32(np.load(f))

        self.modelHist = np.c_[self.modelHist, self.model]
        self.plot_model(ID=0)

    def apply_property(self, n, dataGen):

        # Helpful info: n = element number

        # FIND PROPERTY FOR EACH ELEMENT (FWI - AVERAGE VALUE)
        mu = 0
        if dataGen:
            for node in range(0, 4):
                mu += 0.25*self.realModel[self.mesh.Connect[n, node] - 1, 0]
        if not dataGen:
            for node in range(0, 4):
                mu += 0.25*self.model[self.mesh.Connect[n, node] - 1, 0]

        return mu

    def mount_problem(self, frame, diag_scale, dataGen):

        mass = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)
        m = np.zeros([self.mesh.nNodes, self.mesh.nNodes], dtype=np.float32)

        # Sweeping elements:
        for n in range(0, self.mesh.nElements):

            #FIND PROPERTY FOR EACH ELEMENT (FWI - AVERAGE VALUE)
            mu = 0
            if dataGen:
                for node in range(0, 4):
                    mu += 0.25 * self.realModel[self.mesh.Connect[n, node] - 1, 0]
            if not dataGen:
                for node in range(0, 4):
                    mu += 0.25 * self.model[self.mesh.Connect[n, node] - 1, 0]

            for i in range(0, 4):
                for j in range(0, 4):
                    m[self.mesh.Connect[n, i] - 1, self.mesh.Connect[n, j] - 1] += mu * frame.frameMass[i, j]

        # Lumping mass (row-sum):
        if not diag_scale:
            mass = m.sum(axis=1).reshape((self.mesh.nNodes, 1))

        # Lumping mass (diagonal scaling):
        if diag_scale:
            mtot = m.sum()
            c = mtot / np.trace(m)
            for i in range(0, self.mesh.nNodes):
                mass[i, 0] = c * m[i, i]

        return mass

    def limit(self):

        for node in range(0,self.mesh.nNodes):
            if self.model[node,0] > 1/pow(self.mesh.propSpeed[1],2):
                self.model[node,0] = 1/pow(self.mesh.propSpeed[1],2)
            if self.model[node, 0] < 1 / pow(self.mesh.propSpeed[0], 2):
                self.model[node, 0] = 1 / pow(self.mesh.propSpeed[0], 2)

    def modSens(self,sens,kappa,c,regA,regB,regSens,normSens):

        #Apply dmu/dphi:
        # dmu/dphi = 1 -> sens = sens*1

        #Regularization:
        if regSens:
            sfK = kappa * regA + regB
            sfF = regB.dot(sens)
            sens = scipy.sparse.linalg.spsolve(sfK, sfF).reshape((self.mesh.nNodes, 1))

        #Normalization:
        if normSens:
            maX = np.amax(sens)
            miN = np.amin(sens)
            d = c
            if miN > 0:
                d = maX
            if miN <= 0:
                miN = miN * (-1)
                d = max(miN, maX)
            for node in range(0, self.mesh.nNodes):
                #NOT FILTERING FOR EACH VELOCITY RANGE (NEEDS IMPLEMENTATION)
                sens[node, 0] = c * (sens[node, 0] / d)

        return sens

    def plot_design(self, sources, receivers):

        Sx, Sy, Rx, Ry = [], [], [], []

        for node in range(0, self.mesh.nNodes):
            if node+1 in sources:
                Sx.append(self.mesh.mCoord[node, 0])
                Sy.append(self.mesh.mCoord[node, 1])
            if node+1 in receivers:
                Rx.append(self.mesh.mCoord[node, 0])
                Ry.append(self.mesh.mCoord[node, 1])

        fig1 = plt.figure(figsize=(7, 7))
        plt.scatter(Sx, Sy, color='k', marker='o')
        plt.scatter(Rx, Ry, color='r', marker='x')
        plt.title(' ')
        plt.grid(False)
        plt.xlim([0, self.mesh.lenght])
        plt.ylim([0, self.mesh.depth])
        plt.savefig('../FWI_Python/plots/design/design_layout.png')
        plt.close(fig1)

    def plot_model(self, ID):

        # Helpful info: ID = another code to name the saved im

        aux = np.zeros([self.mesh.nElements, 1])
        axField = np.zeros([self.mesh.nElementsD, self.mesh.nElementsL])

        for e in range(0, self.mesh.nElements):
            counter = 0
            for node in range(0, 4):
                counter += 0.25 * self.model[self.mesh.Connect[e, node] - 1, 0]
            aux[e, 0] = counter

        for j in range(0, self.mesh.nElementsD):
            for i in range(0, self.mesh.nElementsL):
                axField[(self.mesh.nElementsD - 1) - j, i] = aux[i + j * self.mesh.nElementsL, 0]

        #Plot homo_guess
        # if ID == 0:
        #     axField[0,0] = 0.001

        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(axField, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        plt.savefig(f'../FWI_Python/plots/phi_{ID}.png')
        plt.close(fig1)

    def plot_realModel(self):

        # Helpful info: ID = another code to name the saved im

        aux = np.zeros([self.mesh.nElements, 1])
        axField = np.zeros([self.mesh.nElementsD, self.mesh.nElementsL])

        for e in range(0, self.mesh.nElements):
            counter = 0
            for node in range(0, 4):
                counter += 0.25 * self.realModel[self.mesh.Connect[e, node] - 1, 0]
            aux[e, 0] = counter

        for j in range(0, self.mesh.nElementsD):
            for i in range(0, self.mesh.nElementsL):
                axField[(self.mesh.nElementsD - 1) - j, i] = aux[i + j * self.mesh.nElementsL, 0]

        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(axField, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        plt.savefig(f'../FWI_Python/plots/real_phi.png')
        plt.close(fig1)


class MatmodelSIMP(Matmod):

    # 0 <= PHI <= 1

    def __init__(self, meshObj, optConfig):

        super().__init__(meshObj, optConfig)
        self.realModel = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)

    def square_dist(self):

        side = self.mesh.nElementsL // 8
        height = self.mesh.nElementsL // 8
        firstElement = self.mesh.nElementsL // 2 - side // 2 + self.mesh.nElementsL * (
                    self.mesh.nElementsL // 2 - height // 2)
        listOfElements = []

        for i in range(0, height):
            for j in range(0, side):
                listOfElements.append(firstElement + i * self.mesh.nElementsL + j)
        listOfElements.sort()

        for i in range(0, len(listOfElements)):
            for j in range(0, 4):
                self.realModel[self.mesh.Connect[listOfElements[i] - 1, j] - 1, 0] = 1.0

        self.plot_realModel()

    def homoGuess(self):

        self.modelHist = np.c_[self.modelHist, self.model]
        self.plot_model(ID=0)

    def read_RealModel(self):

        with open('../FWI_Python/data_dir/real_model.npy', 'rb') as f:
            self.realModel = np.float32(np.load(f))
        self.plot_realModel()

    def read_Guess(self):

        with open('../FWI_Python/data_dir/guess.npy', 'rb') as f:
            self.model = np.float32(np.load(f))

        self.modelHist = np.c_[self.modelHist, self.model]
        self.plot_model(ID=0)

    def apply_property(self, n, dataGen):

        # Helpful info: n = element number

        # FIND PROPERTY FOR EACH ELEMENT (SIMP - WEIGHTED VALUE)
        mu  = 0
        mu1 = 1 / pow(self.mesh.propSpeed[0], 2)
        mu2 = 1 / pow(self.mesh.propSpeed[1], 2)

        if dataGen:
            for node in range(0, 4):
                mu += 0.25 * (self.realModel[self.mesh.Connect[n, node] - 1, 0] * mu1 +
                              (1 - self.realModel[self.mesh.Connect[n, node] - 1, 0]) * mu2)
        if not dataGen:
            for node in range(0, 4):
                mu += 0.25 * (self.model[self.mesh.Connect[n, node] - 1, 0] * mu1 +
                              (1 - self.model[self.mesh.Connect[n, node] - 1, 0]) * mu2)

        return mu

    def limit(self):

        for node in range(0, self.mesh.nNodes):
            if self.model[node, 0] > 1 :
                self.model[node, 0] = 0.999
            if self.model[node, 0] < 0:
                self.model[node, 0] = 0.001

    def plot_design(self, sources, receivers):

        Sx, Sy, Rx, Ry = [], [], [], []

        for node in range(0, self.mesh.nNodes):
            if node + 1 in sources:
                Sx.append(self.mesh.mCoord[node, 0])
                Sy.append(self.mesh.mCoord[node, 1])
            if node + 1 in receivers:
                Rx.append(self.mesh.mCoord[node, 0])
                Ry.append(self.mesh.mCoord[node, 1])

        fig1 = plt.figure(figsize=(7, 7))
        plt.scatter(Sx, Sy, color='k', marker='o')
        plt.scatter(Rx, Ry, color='r', marker='x')
        plt.title(' ')
        plt.grid(False)
        #plt.axis('square')
        plt.xlim([0, self.mesh.lenght])
        plt.ylim([0, self.mesh.depth])
        plt.savefig('../FWI_Python/plots/design/design_layout.png')
        plt.close(fig1)

    def modSens(self, sens, kappa, c, regA, regB, regSens, normSens):

        # Apply dmu/dphi:
        # dmu/dphi = mu1 - mu2 (NO SIMP PENALIZING -> q = 1)
        mu1 = 1 / pow(self.mesh.propSpeed[0], 2)
        mu2 = 1 / pow(self.mesh.propSpeed[1], 2)
        sens = (mu1 - mu2) * sens

        # Regularization:
        if regSens:
            sfK = kappa * regA + regB
            sfF = regB.dot(sens)
            sens = scipy.sparse.linalg.spsolve(sfK, sfF).reshape((self.mesh.nNodes, 1))

        # Normalization:
        if normSens:
            maX = np.amax(sens)
            miN = np.amin(sens)
            d = c
            if miN > 0:
                d = maX
            if miN <= 0:
                miN = miN * (-1)
                d = max(miN, maX)
            for node in range(0, self.mesh.nNodes):
                # NOT FILTERING FOR EACH VELOCITY RANGE (NEEDS IMPLEMENTATION)
                sens[node, 0] = c * (sens[node, 0] / d)

        return sens

    def plot_model(self, ID):

        # Helpful info: ID = another code to name the saved im

        aux = np.zeros([self.mesh.nElements, 1])
        axField = np.zeros([self.mesh.nElementsD, self.mesh.nElementsL])

        for e in range(0, self.mesh.nElements):
            aux[e, 0] = self.apply_property(e, dataGen=False)

        for j in range(0, self.mesh.nElementsD):
            for i in range(0, self.mesh.nElementsL):
                axField[(self.mesh.nElementsD - 1) - j, i] = aux[i + j * self.mesh.nElementsL, 0]

        if ID == 0:
            axField[0,0] = 0.01

        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(axField, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        plt.savefig(f'../FWI_Python/plots/phi_{ID}.png')
        plt.close(fig1)

    def plot_realModel(self):

        aux = np.zeros([self.mesh.nElements, 1])
        axField = np.zeros([self.mesh.nElementsD, self.mesh.nElementsL])

        for e in range(0, self.mesh.nElements):
            aux[e, 0] = self.apply_property(e, dataGen=True)

        for j in range(0, self.mesh.nElementsD):
            for i in range(0, self.mesh.nElementsL):
                axField[(self.mesh.nElementsD - 1) - j, i] = aux[i + j * self.mesh.nElementsL, 0]

        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(axField, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        plt.savefig(f'../FWI_Python/plots/real_phi.png')
        plt.close(fig1)

    def mount_problem(self, frame, diag_scale, dataGen):

        mass = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)
        m = np.zeros([self.mesh.nNodes, self.mesh.nNodes], dtype=np.float32)

        # Sweeping elements:
        for n in range(0, self.mesh.nElements):

            # FIND PROPERTY FOR EACH ELEMENT (SIMP - WEIGHTED VALUE)
            mu = 0
            mu1 = 1 / pow(self.mesh.propSpeed[0], 2)
            mu2 = 1 / pow(self.mesh.propSpeed[1], 2)

            if dataGen:
                for node in range(0, 4):
                    mu += 0.25 * (self.realModel[self.mesh.Connect[n, node] - 1, 0] * mu1 +
                                  (1 - self.realModel[self.mesh.Connect[n, node] - 1, 0]) * mu2)
            if not dataGen:
                for node in range(0, 4):
                    mu += 0.25 * (self.model[self.mesh.Connect[n, node] - 1, 0] * mu1 +
                                  (1 - self.model[self.mesh.Connect[n, node] - 1, 0]) * mu2)

            for i in range(0, 4):
                for j in range(0, 4):
                    m[self.mesh.Connect[n, i] - 1, self.mesh.Connect[n, j] - 1] += mu * frame.frameMass[i, j]

        # Lumping mass (row-sum):
        if not diag_scale:
            mass = m.sum(axis=1).reshape((self.mesh.nNodes, 1))

        # Lumping mass (diagonal scaling):
        if diag_scale:
            mtot = m.sum()
            c = mtot / np.trace(m)
            for i in range(0, self.mesh.nNodes):
                mass[i, 0] = c * m[i, i]

        return mass


class MatmodelLS(Matmod):

    # -1 <= PHI <= 1

    def __init__(self, meshObj, optConfig):

        super().__init__(meshObj, optConfig)
        self.realModel = np.ones([self.mesh.nNodes, 1], dtype=np.float32) * -0.01
        self.delta *= 2.5

    def square_dist(self):

        side = self.mesh.nElementsL // 6
        height = self.mesh.nElementsL // 6
        firstElement = self.mesh.nElementsL // 2 - side // 2 + self.mesh.nElementsL * (
                    self.mesh.nElementsL // 2 - height // 2)
        listOfElements = []

        for i in range(0, height):
            for j in range(0, side):
                listOfElements.append(firstElement + i * self.mesh.nElementsL + j)
        listOfElements.sort()

        for i in range(0, len(listOfElements)):
            for j in range(0, 4):
                self.realModel[self.mesh.Connect[listOfElements[i] - 1, j] - 1, 0] = 0.01

        self.plot_realModel()

    def homoGuess(self):

        self.model[:, 0] = -0.01
        self.modelHist = np.c_[self.modelHist, self.model]

        self.plot_model(ID=0)

    def read_RealModel(self):

        with open('../FWI_Python/data_dir/real_model.npy', 'rb') as f:
            self.realModel = np.float32(np.load(f))
        self.plot_realModel()

    def read_Guess(self):

        with open('../FWI_Python/data_dir/guess.npy', 'rb') as f:
            self.model = np.float32(np.load(f))

        self.modelHist = np.c_[self.modelHist, self.model]
        self.plot_model(ID=0)

    def apply_property(self, n, dataGen):

        # Helpful info: n = element number

        # FIND PROPERTY FOR ELEMENT (LEVEL SET - FIT CURVE)
        phiElement = np.zeros([4, 1], dtype=np.float32)
        if dataGen:
            for node in range(0, 4):
                phiElement[node, 0] = self.realModel[self.mesh.Connect[n, node] - 1, 0]

        if not dataGen:
            for node in range(0, 4):
                phiElement[node, 0] = self.model[self.mesh.Connect[n, node] - 1, 0]

        fit = CurveFitting(phiElement)
        fit.fitLevelSet(order=2, gridSize=5)
        mu = fit.distribute(1 / pow(self.mesh.propSpeed[0], 2), 1 / pow(self.mesh.propSpeed[1], 2))

        return mu

    def limit(self):

        for node in range(0, self.mesh.nNodes):
            if self.model[node, 0] > 1:
                self.model[node, 0] = 0.999
            if self.model[node, 0] < -1:
                self.model[node, 0] = -0.999

    def plot_design(self, sources, receivers):

        Sx, Sy, Rx, Ry = [], [], [], []

        for node in range(0, self.mesh.nNodes):
            if node + 1 in sources:
                Sx.append(self.mesh.mCoord[node, 0])
                Sy.append(self.mesh.mCoord[node, 1])
            if node + 1 in receivers:
                Rx.append(self.mesh.mCoord[node, 0])
                Ry.append(self.mesh.mCoord[node, 1])

        fig1 = plt.figure(figsize=(7, 7))
        plt.scatter(Sx, Sy, color='k', marker='o')
        plt.scatter(Rx, Ry, color='r', marker='x')
        plt.title(' ')
        plt.grid(False)
        #plt.axis('square')
        plt.xlim([0, self.mesh.lenght])
        plt.ylim([0, self.mesh.depth])
        plt.savefig('../FWI_Python/plots/design/design_layout.png')
        plt.close(fig1)

    def modSens(self, sens, kappa, c, regA, regB, regSens, normSens):

        # Apply dmu/dphi:
        # dmu/dphi = mu1 - mu2 (DELTA DIRAC SUPPRESSED)
        mu1 = 1 / pow(self.mesh.propSpeed[0], 2)
        mu2 = 1 / pow(self.mesh.propSpeed[1], 2)
        sens = (mu1 - mu2) * sens
        # if 0 < phi < 2:
        #     sens1 = (mu1 - mu2) * sens
        # if 1 < phi < 3:
        #     sens2 = (mu2 - mu3) * sens
        # sens = sens1 + sens2

        # Regularization:
        if regSens:
            sfK = kappa * regA + regB
            sfF = regB.dot(sens)
            sens = scipy.sparse.linalg.spsolve(sfK, sfF).reshape((self.mesh.nNodes, 1))

        # Normalization:
        if normSens:
            maX = np.amax(sens)
            miN = np.amin(sens)
            d = c
            if miN > 0:
                d = maX
            if miN <= 0:
                miN = miN * (-1)
                d = max(miN, maX)
            for node in range(0, self.mesh.nNodes):
                # NOT FILTERING FOR EACH VELOCITY RANGE (NEEDS IMPLEMENTATION)
                sens[node, 0] = c * (sens[node, 0] / d)

        return sens

    def plot_model(self, ID):

        # Helpful info: ID = another code to name the saved im

        aux = np.zeros([self.mesh.nElements, 1])
        axField = np.zeros([self.mesh.nElementsD, self.mesh.nElementsL])

        for e in range(0, self.mesh.nElements):
            aux[e, 0] = self.apply_property(e, dataGen=False)

        for j in range(0, self.mesh.nElementsD):
            for i in range(0, self.mesh.nElementsL):
                axField[(self.mesh.nElementsD - 1) - j, i] = aux[i + j * self.mesh.nElementsL, 0]

        #Adjust colormap center to zero:
        # vmini = 0
        # vmaxi = 2
        # mini = abs(np.amin(axField))
        # maxi = abs(np.amax(axField))
        # if maxi > mini:
        #     vmaxi = maxi
        #     vmini = -maxi
        # elif maxi < mini:
        #     vmaxi = mini
        #     vmini = -mini
        # elif maxi == mini:
        #     vmaxi = maxi
        #     vmini = -mini

        if ID == 0:
            axField[0,0] = 0.0

        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(axField, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        plt.savefig(f'../FWI_Python/plots/phi_{ID}.png')
        plt.close(fig1)

    def plot_realModel(self):

        aux = np.zeros([self.mesh.nElements, 1])
        axField = np.zeros([self.mesh.nElementsD, self.mesh.nElementsL])

        for e in range(0, self.mesh.nElements):
            aux[e, 0] = self.apply_property(e, dataGen=True)

        for j in range(0, self.mesh.nElementsD):
            for i in range(0, self.mesh.nElementsL):
                axField[(self.mesh.nElementsD - 1) - j, i] = aux[i + j * self.mesh.nElementsL, 0]

        # Adjust colormap center to zero:
        # vmini = 0
        # vmaxi = 2
        # mini = abs(np.amin(axField))
        # maxi = abs(np.amax(axField))
        # if maxi > mini:
        #     vmaxi = maxi
        #     vmini = -maxi
        # elif maxi < mini:
        #     vmaxi = mini
        #     vmini = -mini
        # elif maxi == mini:
        #     vmaxi = maxi
        #     vmini = -mini

        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(axField, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        plt.savefig(f'../FWI_Python/plots/real_phi.png')
        plt.close(fig1)

    def mount_problem(self, frame, diag_scale, dataGen):

        mass = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)
        m = np.zeros([self.mesh.nNodes, self.mesh.nNodes], dtype=np.float32)
        phiElement = np.zeros([4, 1], dtype=np.float32)

        # Sweeping elements:
        for n in range(0, self.mesh.nElements):

            # FIND PROPERTY FOR ELEMENT (LEVEL SET - FIT CURVE)
            if dataGen:
                for node in range(0, 4):
                    phiElement[node, 0] = self.realModel[self.mesh.Connect[n, node] - 1, 0]

            if not dataGen:
                for node in range(0, 4):
                    phiElement[node, 0] = self.model[self.mesh.Connect[n, node] - 1, 0]

            fit = CurveFitting(phiElement)
            fit.fitLevelSet(order=2, gridSize=5)
            mu = fit.distribute(1 / pow(self.mesh.propSpeed[0], 2), 1 / pow(self.mesh.propSpeed[1], 2))

            for i in range(0, 4):
                for j in range(0, 4):
                    m[self.mesh.Connect[n, i] - 1, self.mesh.Connect[n, j] - 1] += mu * frame.frameMass[i, j]

        # Lumping mass (row-sum):
        if not diag_scale:
            mass = m.sum(axis=1).reshape((self.mesh.nNodes, 1))

        # Lumping mass (diagonal scaling):
        if diag_scale:
            mtot = m.sum()
            c = mtot / np.trace(m)
            for i in range(0, self.mesh.nNodes):
                mass[i, 0] = c * m[i, i]

        return mass


class MatmodelMMLS(Matmod):

    # 0 <= PHI <= n materials

    def __init__(self, meshObj, optConfig):

        super().__init__(meshObj, optConfig)
        self.realModel = np.ones([self.mesh.nNodes, 1], dtype=np.float32)
        self.delta *= 2.5

        self.velocities = 0
        self.nM = 0
        self.mus = []


    def apriori_Velocities(self,velocities):

        self.velocities = velocities
        self.nM = len(self.velocities)
        self.mus = [1/pow(self.velocities[i], 2) for i in range(0, self.nM)]

    def homoGuess(self):

        self.model[:, 0] = 0.00001
        self.modelHist = np.c_[self.modelHist, self.model]

        self.plot_model(ID=0)

    def read_RealModel(self):

        with open('../FWI_Python/data_dir/real_model.npy', 'rb') as f:
            self.realModel = np.float32(np.load(f))
        self.plot_realModel()

    def read_Guess(self):

        with open('../FWI_Python/data_dir/guess.npy', 'rb') as f:
            self.model = np.float32(np.load(f))

        self.modelHist = np.c_[self.modelHist, self.model]
        self.plot_model(ID=0)

    def strat_guess(self):

        for i in range(0,self.mesh.nNodes):
            self.model[i,0] = i*(self.nM/self.mesh.nNodes)

        self.modelHist = np.c_[self.modelHist, self.model]
        self.plot_model(ID=0)

    def apply_property(self, n, dataGen):

        # Helpful info: n = element number
        avg = 0
        l = []

        if dataGen:
            for node in range(0, 4):
                l.append(int(self.realModel[self.mesh.Connect[n, node] - 1, 0]))
                avg += 0.25 * self.realModel[self.mesh.Connect[n, node] - 1, 0]

        if not dataGen:
            for node in range(0, 4):
                l.append(int(self.model[self.mesh.Connect[n, node] - 1, 0]))
                avg += 0.25 * self.model[self.mesh.Connect[n,node]-1,0]

        a = l.count(l[0])
        b = 4 - a

        if l[0] == max(l):
            mu = (a/4)*self.mus[l[0]] + (b/4)*self.mus[l[0]-1]
        else:
            mu = (a/4) * self.mus[l[0]] + (b/4) * self.mus[l[0]+1]

        return mu

    def limit(self):

        for node in range(0, self.mesh.nNodes):
            if self.model[node, 0] > self.nM:
                self.model[node, 0] = (self.nM-1) * 0.9999999
            if self.model[node, 0] < 0:
                self.model[node, 0] = 0.00001

    def plot_design(self, sources, receivers):

        Sx, Sy, Rx, Ry = [], [], [], []

        for node in range(0, self.mesh.nNodes):
            if node+1 in sources:
                Sx.append(self.mesh.mCoord[node, 0])
                Sy.append(self.mesh.mCoord[node, 1])
            if node+1 in receivers:
                Rx.append(self.mesh.mCoord[node, 0])
                Ry.append(self.mesh.mCoord[node, 1])

        fig1 = plt.figure(figsize=(7, 7))
        plt.scatter(Sx, Sy, color='k', marker='o')
        plt.scatter(Rx, Ry, color='r', marker='x')
        plt.title(' ')
        plt.grid(False)
        plt.savefig('../FWI_Python/plots/design/design_layout.png')
        plt.close(fig1)

    def modSens(self, sens, kappa, c, regA, regB, regSens, normSens):

        # Apply dmu/dphi:
        H = np.zeros((self.mesh.nNodes,1),dtype=np.float32)

        for k in range(1,self.nM):
            H[np.where(np.logical_and(self.model <= k+1, self.model >= k-1))] += self.mus[k] - self.mus[k-1]

        sens = np.multiply(H,sens)

        # Regularization:
        if regSens:
            sfK = kappa * regA + regB
            sfF = regB.dot(sens)
            sens = scipy.sparse.linalg.spsolve(sfK, sfF).reshape((self.mesh.nNodes, 1))

        # Normalization:
        if normSens:
            maX = np.amax(sens)
            miN = np.amin(sens)
            d = c
            if miN > 0:
                d = maX
            if miN <= 0:
                miN = miN * (-1)
                d = max(miN, maX)
            for node in range(0, self.mesh.nNodes):
                # NOT FILTERING FOR EACH VELOCITY RANGE (NEEDS IMPLEMENTATION)
                sens[node, 0] = c * (sens[node, 0] / d)

        return sens

    def plot_model(self, ID):

        # Helpful info: ID = another code to name the saved im

        aux = np.zeros([self.mesh.nElements, 1])
        axField = np.zeros([self.mesh.nElementsD, self.mesh.nElementsL])

        for e in range(0, self.mesh.nElements):
            aux[e, 0] = self.apply_property(e, dataGen=False)

        for j in range(0, self.mesh.nElementsD):
            for i in range(0, self.mesh.nElementsL):
                axField[(self.mesh.nElementsD - 1) - j, i] = aux[i + j * self.mesh.nElementsL, 0]

        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(axField, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        plt.savefig(f'../FWI_Python/plots/phi_{ID}.png')
        plt.close(fig1)

    def plot_realModel(self):

        aux = np.zeros([self.mesh.nElements, 1])
        axField = np.zeros([self.mesh.nElementsD, self.mesh.nElementsL])

        for e in range(0, self.mesh.nElements):
            aux[e, 0] = self.apply_property(e, dataGen=True)

        for j in range(0, self.mesh.nElementsD):
            for i in range(0, self.mesh.nElementsL):
                axField[(self.mesh.nElementsD - 1) - j, i] = aux[i + j * self.mesh.nElementsL, 0]

        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(axField, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        plt.savefig(f'../FWI_Python/plots/real_phi.png')
        plt.close(fig1)

    def mount_problem(self, frame, diag_scale, dataGen):

        mass = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)
        m = np.zeros([self.mesh.nNodes, self.mesh.nNodes], dtype=np.float32)
        phiElement = np.zeros([4, 1], dtype=np.float32)

        # Sweeping elements:
        for n in range(0, self.mesh.nElements):

            # FIND PROPERTY FOR ELEMENT (LEVEL SET - FIT CURVE)
            avg = 0

            if dataGen:
                for node in range(0, 4):
                    avg += 0.25 * self.realModel[self.mesh.Connect[n, node] - 1, 0]

            if not dataGen:
                for node in range(0, 4):
                    avg += 0.25 * self.model[self.mesh.Connect[n, node] - 1, 0]

            mu = self.mus[int(avg)]

            for i in range(0, 4):
                for j in range(0, 4):
                    m[self.mesh.Connect[n, i] - 1, self.mesh.Connect[n, j] - 1] += mu * frame.frameMass[i, j]

        # Lumping mass (row-sum):
        if not diag_scale:
            mass = m.sum(axis=1).reshape((self.mesh.nNodes, 1))

        # Lumping mass (diagonal scaling):
        if diag_scale:
            mtot = m.sum()
            c = mtot / np.trace(m)
            for i in range(0, self.mesh.nNodes):
                mass[i, 0] = c * m[i, i]

        return mass