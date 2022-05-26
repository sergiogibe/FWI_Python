import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from curveFitting import CurveFitting
import scipy.sparse
import scipy.sparse.linalg


class Matmod:

    def __init__(self, meshObj, optConfig):

        self.mesh        = meshObj
        self.delta       = optConfig[0]
        self.reset_delta = optConfig[0]
        self.niter       = optConfig[1]

        self.model       = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)
        self.modelHist   = np.zeros([self.mesh.nNodes, 0], dtype=np.float32)

        self.velocities = []

    # SUPER CLASS METHODS
    def writeHist(self):
        self.modelHist = np.c_[self.modelHist, self.model]

    def update_reactdiff(self, sens, difA, difB):
        #Reaction-Diffusion Evolution
        lsK = difA + (1/self.delta)*difB
        lsF = difB.dot((1/self.delta)*self.model.reshape((self.mesh.nNodes,1)) + sens.reshape((self.mesh.nNodes,1)))

        self.model = scipy.sparse.linalg.spsolve(lsK,lsF).reshape((self.mesh.nNodes,1))

    def update_gradient(self,sens):
        # TODO: Fix Steepest decent evolution
        self.model += sens

    def control_step(self,costFunctionNew,costFunctionOld,iteration,boost,manualReset):
        # This one waits for the third iteration.
        if iteration > 2 and costFunctionNew / costFunctionOld < 1:
            self.delta *= costFunctionNew / costFunctionOld
            self.delta *= boost
        if manualReset == True and iteration % 6 == 0 and iteration > 2:
            print('Reset step? 1- yes   2- no')
            x = int(input())
            if x == 1:
                self.reset_step()

    def reset_step(self):
        self.delta = self.reset_delta


class MatmodelFWI(Matmod):

    # mu1 <= PHI <= mu2

    def __init__(self, meshObj, optConfig):

        super().__init__(meshObj, optConfig)
        self.realModel      = np.ones([self.mesh.nNodes, 1], dtype=np.float32) * (1/pow(self.mesh.propSpeed[1],2))
        self.delta *= 5 #NORMALLY FWI STEP NEEDS TO BE BIGGER

        # TODO: Fix FWI.
        print('FWI material model not working yet.')
        exit()

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

        with open('../../FWI_Python/data_dir/real_model.npy', 'rb') as f:
            self.realModel = np.float32(np.load(f))
        self.plot_realModel()

    def read_Guess(self):

        with open('../../FWI_Python/data_dir/guess.npy', 'rb') as f:
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
        plt.savefig('../../FWI_Python/plots/design/design_layout.png')
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
        plt.savefig(f'../../FWI_Python/plots/phi_{ID}.png')
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
        plt.savefig(f'../../FWI_Python/plots/real_phi.png')
        plt.close(fig1)


class MatmodelSIMP(Matmod):

    # 0 <= PHI <= 1

    def __init__(self, meshObj, optConfig):

        super().__init__(meshObj, optConfig)
        self.realModel = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)

        # TODO: Fix SIMP.
        print('SIMP material model not working yet.')
        exit()

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

    def __init__(self, meshObj, optConfig):

        super().__init__(meshObj, optConfig)
        self.realModel = np.ones([self.mesh.nNodes, 1], dtype=np.float32) * -0.01
        self.delta *= 2.5

    def set_velocities(self,velocities):

        self.velocities = velocities
        if len(velocities) > 2:
            print('Level set only accounts for 2 materials. For more materials try MMLS.')
            exit()
        self.mus = [1/pow(self.velocities[i], 2) for i in range(0, 2)]

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

        with open('../../FWI_Python/data_dir/real_model.npy', 'rb') as f:
            self.realModel = np.float32(np.load(f))
        self.plot_realModel()

    def read_Guess(self):

        with open('../../FWI_Python/data_dir/guess.npy', 'rb') as f:
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
        mu = fit.distribute(self.mus[1], self.mus[0])

        return mu

    def limit(self):

        for node in range(0, self.mesh.nNodes):
            if self.model[node, 0] > 1:
                self.model[node, 0] = 0.999
            if self.model[node, 0] < -1:
                self.model[node, 0] = -0.999

    def plot_design(self, sources, receivers):

        # TODO: Improve the accuracy of this plot and update other material models.

        Sx, Sy, Rx, Ry = [], [], [], []

        for node in range(0, self.mesh.nNodes):
            if node + 1 in sources:
                Sx.append((self.mesh.mCoord[node, 0]/self.mesh.length)*self.mesh.nElementsL)
                Sy.append((self.mesh.mCoord[node, 1]/self.mesh.depth)*self.mesh.nElementsD)
            if node + 1 in receivers:
                Rx.append((self.mesh.mCoord[node, 0]/self.mesh.length)*self.mesh.nElementsL)
                Ry.append((self.mesh.mCoord[node, 1]/self.mesh.depth)*self.mesh.nElementsD)

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

        plt.scatter(Sx, Sy, color='k', marker='o')
        plt.scatter(Rx, Ry, color='k', marker='x')
        plt.title(' ')
        plt.grid(False)

        plt.savefig('../../FWI_Python/plots/design/design_layout.png')
        plt.close(fig1)

    def modSens(self, sens, kappa, c, regA, regB, regSens, normSens):

        # Apply dmu/dphi:
        # dmu/dphi = mu1 - mu2 (DELTA DIRAC SUPPRESSED)
        sens = (self.mus[1] - self.mus[0]) * sens

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

        aux = np.zeros([self.mesh.nElements, 1])
        axField = np.zeros([self.mesh.nElementsD, self.mesh.nElementsL])

        for e in range(0, self.mesh.nElements):
            aux[e, 0] = self.apply_property(e, dataGen=False)

        for j in range(0, self.mesh.nElementsD):
            for i in range(0, self.mesh.nElementsL):
                axField[(self.mesh.nElementsD - 1) - j, i] = aux[i + j * self.mesh.nElementsL, 0]

        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(axField, vmin=self.mus[1], vmax=self.mus[0], cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        plt.savefig(f'../../FWI_Python/plots/phi_{ID}.png')
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
        plt.savefig(f'../../FWI_Python/plots/real_phi.png')
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
            mu = fit.distribute(self.mus[1], self.mus[0])

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

    def __init__(self, meshObj, optConfig):

        super().__init__(meshObj, optConfig)
        self.realModel = np.ones([self.mesh.nNodes, 1], dtype=np.float32)
        self.delta *= 1.5

        self.nM = 0
        self.mus = []

    def set_velocities(self,velocities):

        self.velocities = velocities
        self.nM = len(self.velocities)
        self.mus = [1/pow(self.velocities[i], 2) for i in range(0, self.nM)]

    def square_dist(self,sideDiv,value,plt):

        side = self.mesh.nElementsL // sideDiv
        height = self.mesh.nElementsL // sideDiv
        firstElement = self.mesh.nElementsL // 2 - side // 2 + self.mesh.nElementsL * (
                    self.mesh.nElementsL // 2 - height // 2)
        listOfElements = []

        for i in range(0, height):
            for j in range(0, side):
                listOfElements.append(firstElement + i * self.mesh.nElementsL + j)
        listOfElements.sort()

        for i in range(0, len(listOfElements)):
            for j in range(0, 4):
                self.realModel[self.mesh.Connect[listOfElements[i] - 1, j] - 1, 0] = value

        if plt:
            self.plot_realModel()

    def homoGuess(self):

        self.model[:, 0] = 0.99
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

    def strat_model(self):

        for i in range(0,self.mesh.nNodes):
            self.realModel[i,0] = i*(self.nM/self.mesh.nNodes)

        self.plot_realModel()

    def strat_guess(self):

        # TODO: Fix this one (not correct i think)
        print('Strat guess not implemented yet.')
        exit()

        for i in range(0,self.mesh.nNodes):
            self.model[i,0] = i*(self.nM/self.mesh.nNodes)

        self.modelHist = np.c_[self.modelHist, self.model]
        self.plot_model(ID=0)

    def apply_property(self, n, dataGen):

        # Helpful info: This function is used to plot only.
        #               Function's context: inside the element n

        avg = 0

        if dataGen:
            for node in range(0, 4):
                avg += 0.25 * self.realModel[self.mesh.Connect[n, node] - 1, 0]

        if not dataGen:
            for node in range(0, 4):
                avg += 0.25 * self.model[self.mesh.Connect[n,node]-1,0]

        # Example with 4 materials:
        # 1 material - control function interval = [0,1]
        # 2 material - control function interval = [1,2]
        # 3 material - control function interval = [2,3]
        # 4 material - control function interval = [3,4]

        material = int(avg)
        if material >= self.nM:
            material = self.nM - 1
        if material <= 0:
            material = 0
        mu = self.mus[material]

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
                Sx.append((self.mesh.mCoord[node, 0] / self.mesh.lenght) * self.mesh.nElementsL)
                Sy.append((self.mesh.mCoord[node, 1] / self.mesh.depth) * self.mesh.nElementsD)
            if node+1 in receivers:
                Rx.append((self.mesh.mCoord[node, 0] / self.mesh.lenght) * self.mesh.nElementsL)
                Ry.append((self.mesh.mCoord[node, 1] / self.mesh.depth) * self.mesh.nElementsD)

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

        plt.scatter(Sx, Sy, color='k', marker='o')
        plt.scatter(Rx, Ry, color='k', marker='x')
        plt.title(' ')
        plt.grid(False)

        plt.savefig('../../FWI_Python/plots/design/design_layout.png')
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
                # TODO: FILTER EACH VELOCITY RANGE
                sens[node, 0] = c * (sens[node, 0] / d)

        return sens

    def plot_model(self, ID):

        aux = np.zeros([self.mesh.nElements, 1])
        axField = np.zeros([self.mesh.nElementsD, self.mesh.nElementsL])

        for e in range(0, self.mesh.nElements):
            aux[e, 0] = self.apply_property(e, dataGen=False)

        for j in range(0, self.mesh.nElementsD):
            for i in range(0, self.mesh.nElementsL):
                axField[(self.mesh.nElementsD - 1) - j, i] = aux[i + j * self.mesh.nElementsL, 0]

        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(axField, vmin=self.mus[self.nM-1], vmax=self.mus[0], cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        # TODO: fig1.colorbar(cm.ScalarMappable(cmap='seismic'), ax=ax)
        plt.savefig(f'../../FWI_Python/plots/phi_{ID}.png')
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
        ax.imshow(axField, vmin=self.mus[self.nM-1], vmax=self.mus[0], cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        plt.savefig(f'../../FWI_Python/plots/real_phi.png')
        plt.close(fig1)

    def mount_problem(self, frame, diag_scale, dataGen):

        mass = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)
        m = np.zeros([self.mesh.nNodes, self.mesh.nNodes], dtype=np.float32)
        #phiElement = np.zeros([4, 1], dtype=np.float32)

        # Sweeping elements:
        for n in range(0, self.mesh.nElements):

            # TODO: FIND PROPERTY FOR ELEMENT (LEVEL SET - FIT CURVE)
            avg = 0

            if dataGen:
                for node in range(0, 4):
                    avg += 0.25 * self.realModel[self.mesh.Connect[n, node] - 1, 0]

            if not dataGen:
                for node in range(0, 4):
                    avg += 0.25 * self.model[self.mesh.Connect[n, node] - 1, 0]

            # TODO: Apply cross-material.
            material = int(avg)
            if material >= self.nM:
                material = self.nM - 1
            if material <= 0:
                material = 0
            mu = self.mus[material]

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


#=====================================


class MultiControl:

    def __init__(self, meshObj, optConfig):

        self.mesh        = meshObj
        self.delta       = optConfig[0]
        self.reset_delta = optConfig[0]
        self.niter       = optConfig[1]

        self.delta      *= 1.5

        # Only 3 materials.
        self.model1      = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)
        self.model2      = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)

        self.modelHist1  = np.zeros([self.mesh.nNodes, 0], dtype=np.float32)
        self.modelHist2  = np.zeros([self.mesh.nNodes, 0], dtype=np.float32)

    def writeHist(self):
        self.modelHist1 = np.c_[self.modelHist1, self.model1]
        self.modelHist2 = np.c_[self.modelHist2, self.model2]

    def update_reactdiff(self, sens1, sens2, difA, difB):
        # Update first control function
        lsK = difA + (1/self.delta)*difB
        lsF = difB.dot((1/self.delta)*self.model1.reshape((self.mesh.nNodes,1)) + sens1.reshape((self.mesh.nNodes,1)))
        self.model1 = scipy.sparse.linalg.spsolve(lsK,lsF).reshape((self.mesh.nNodes,1))

        # Update second control function
        lsK = difA + (1 / self.delta) * difB
        lsF = difB.dot((1 / self.delta) * self.model2.reshape((self.mesh.nNodes, 1)) + sens2.reshape((self.mesh.nNodes, 1)))
        self.model2 = scipy.sparse.linalg.spsolve(lsK, lsF).reshape((self.mesh.nNodes, 1))

    def control_step(self,costFunctionNew,costFunctionOld,iteration,boost,manualReset):
        # This one waits for the third iteration.
        if iteration > 2 and costFunctionNew / costFunctionOld < 1:
            self.delta *= costFunctionNew / costFunctionOld
            self.delta *= boost
        if manualReset == True and iteration % 6 == 0 and iteration > 2:
            print('Reset step? 1- yes   2- no')
            x = int(input())
            if x == 1:
                self.reset_step()

    def reset_step(self):
        self.delta = self.reset_delta

    def set_velocities(self,velocities):

        self.velocities = velocities
        if len(velocities) != 3:
            print('Multicontrol implemented for 3 materials. ')
            exit()
        self.mus = [1/pow(self.velocities[i], 2) for i in range(0, len(velocities))]

    def homoGuess(self):

        self.model1[:, 0] = -0.01
        self.model2[:, 0] = -0.01
        self.modelHist1 = np.c_[self.modelHist1, self.model1]
        self.modelHist2 = np.c_[self.modelHist2, self.model2]

        #self.plot_model(ID=0)

    def apply_property(self, n):

        # Function's context: inside the element n

        avg1, avg2 = 0, 0
        mu = 100              # it doesnt matter

        for node in range(0, 4):
            avg1 += 0.25 * self.model1[self.mesh.Connect[n, node] - 1, 0]
            avg2 += 0.25 * self.model2[self.mesh.Connect[n, node] - 1, 0]

        if avg1 >= 0:
            if avg2 < 0:
                mu = self.mus[1]
            else:
                mu = self.mus[2]
        else:
            mu = self.mus[0]

        return mu

    def modSens(self, sens, kappa, c, regA, regB, whichModel, regSens, normSens):

        # Apply dmu/dphi_i (material model - multicontrol)
        if whichModel == 1:
            for n in range(0,self.mesh.nNodes):
                a = 1
                if self.model2[n,0] >= 0:
                    a = 0
                sens[n,0] = (self.mus[1]*a - self.mus[0]) * sens[n,0]
        elif whichModel == 2:
            for n in range(0, self.mesh.nNodes):
                a = 0
                if self.model1[n, 0] >= 0:
                    a = 1
                sens[n, 0] = (self.mus[2] - a*self.mus[1]) * sens[n, 0]
        else:
            print('Multicontrol implemented for 2 models (control functions) only. ')
            exit()

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
                sens[node, 0] = c * (sens[node, 0] / d)

        return sens

    def plot_model(self, ID):

        aux = np.zeros([self.mesh.nElements, 1])
        axField = np.zeros([self.mesh.nElementsD, self.mesh.nElementsL])

        for e in range(0, self.mesh.nElements):
            aux[e, 0] = self.apply_property(e)

        for j in range(0, self.mesh.nElementsD):
            for i in range(0, self.mesh.nElementsL):
                axField[(self.mesh.nElementsD - 1) - j, i] = aux[i + j * self.mesh.nElementsL, 0]

        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(axField, vmin=self.mus[2], vmax=self.mus[0], cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        plt.savefig(f'../../FWI_Python/plots/phi_{ID}.png')
        plt.close(fig1)

    def mount_problem(self, frame, diag_scale):

        mass = np.zeros([self.mesh.nNodes, 1], dtype=np.float32)
        m = np.zeros([self.mesh.nNodes, self.mesh.nNodes], dtype=np.float32)
        phiElement = np.zeros([4, 1], dtype=np.float32)

        # Sweeping elements:
        for n in range(0, self.mesh.nElements):
            mu = self.apply_property(n)

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