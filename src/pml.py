import numpy as np
from plot import *

class PML:
    def __init__(self, mesh, thickness, atten):

        self.mesh  = mesh
        self.atten = atten             # atten = 1 => no PML
        self.e     = thickness         # (thickness in nodes)
        self.forcePML = np.ones((self.mesh.nNodes), dtype=np.float32)
        nl = mesh.nElementsL + 1
        nd = mesh.nElementsD + 1

        for ie in range(self.e,0,-1):

            pmlList = []

            # BOTTOM
            for j in range(1, ie+1):
                for i in range(1, nl+1):
                    pmlList.append(i+(j-1)*nl)

            # TOP
            siz = len(pmlList)
            for i in range(mesh.nNodes-siz+1,mesh.nNodes+1):
                pmlList.append(i)

            # LEFT
            for j in range(1, nd+1):
                for i in range(1,ie+1):
                    pmlList.append(i+(j-1)*nl)

            # RIGHT
            for j in range(1, nd+1):
                for i in range(nl-ie+1,nl+1):
                    pmlList.append(i + (j - 1) * nl)

            # REMOVE DUPLICATES, SORT AND MAKE ARRAY
            pmlList = list(dict.fromkeys(pmlList))
            pmlList.sort()
            self.pml = np.asarray(pmlList,dtype=np.int32)   # array with attenuation nodes only

            # GET FORCE PML WITH ATTENUATION
            for i in range(self.pml.shape[0]):
                self.forcePML[self.pml[i]-1] = self.atten*(1 - 1/pow(ie,0.8))


    def plot_PML(self):
        #plot_inDomain(self.pml, self.mesh, "pml_check", "0")
        nNodesL = self.mesh.nElementsL + 1
        nNodesD = self.mesh.nElementsD + 1

        axField = np.zeros([nNodesD, nNodesL])

        for j in range(0, nNodesD):
            for i in range(0, nNodesL):
                axField[(nNodesD - 1) - j, i] = self.forcePML[i + j * nNodesL]

        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(axField, cmap='binary')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.title(' ')
        plt.savefig(f'../../FWI_Python/plots/check_PML_0.png')
        plt.close(fig1)