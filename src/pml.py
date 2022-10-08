import numpy as np
from .plot import *
import os


class PML:
    def __init__(self, 
                 mesh: object, 
                 thickness: int, 
                 atten: int) -> None:
        """   This class is responsible for:
           - Creating the absorbing layers
           - It also plots the final layout
        """

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
                self.forcePML[self.pml[i]-1] = 1 - pow((self.e-ie)/self.e,self.atten) #pow(self.atten,(self.e-ie)*0.1)


        nNodesL = self.mesh.nElementsL + 1
        nNodesD = self.mesh.nElementsD + 1

        axField = np.zeros([nNodesD, nNodesL])

        for j in range(0, nNodesD):
            for i in range(0, nNodesL):
                axField[(nNodesD - 1) - j, i] = self.forcePML[i + j * nNodesL]


        fig1, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(np.transpose(axField), cmap='binary_r')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        #plt.xlim(0.0, self.mesh.length)
        #plt.ylim(self.mesh.depth, 0.0)
        plt.xlabel('X (km)')
        plt.ylabel('Z (km)')
        plt.gca().set_aspect('equal')
        plt.savefig(f'{os.getcwd()}/plots/PML_Layout.png')
        plt.close(fig1)