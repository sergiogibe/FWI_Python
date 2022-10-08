import numpy as np
from plot import plot_inDomain

def nodalPos(pos: list, mesh: object):
    '''INPUT: pos = [ (x1,y1), (x2,y2), (x3,y3) ... (xn,yn) ] for n sources or receivers'''

    posList = []
    for position in pos:
        side_ratio = position[0] / mesh.length
        if side_ratio > 1.0:
            side_ratio = 1.0
        if side_ratio < 0.0:
            side_ratio = 0.0
        nodalX = 1 + round(side_ratio * mesh.nElementsL)
        if nodalX == 0:
            nodalX = 1
        side_ratio = position[1] / mesh.depth
        if side_ratio > 1.0:
            side_ratio = 1.0
        if side_ratio < 0.0:
            side_ratio = 0.0
        nodalY = 1 + round(side_ratio * mesh.nElementsD)
        if nodalY == 0:
            nodalY = 1
        nodalAbsolute = nodalX + (mesh.nElementsL + 1) * (nodalY - 1)
        posList.append(nodalAbsolute)  # Nodal positions (1 for the first node and so on)
    posList.sort()

    return np.asarray(posList, dtype=np.int32)

def make_inclusions(incs: list, mesh: object, control: np.array, value:float):
    '''INPUT: pos = [ (xi1,xf1,yi1,yf1) ... (xin,xfn,yin,yfn) ] for n inclusions'''

    nl = mesh.nElementsL+1
    nd = mesh.nElementsD+1

    for rec in incs:
        nodesList = []

        xi: int = round((rec[0] * nl) / mesh.length)
        xf: int = round((rec[1] * nl) / mesh.length)
        yi: int = round((rec[2] * nd)  / mesh.depth)
        yf: int = round((rec[3] * nd)  / mesh.depth)

        # FIRST ABSOLUTE NODE | THICKNESS | HEIGHT :
        n1 = xi + yi * nl
        t  = xf - xi
        h  = yf - yi
        #  00000000000000000
        #  000 n1------- 000
        #  000 --------- 000
        #  000 --------- 000
        #  000 --------- 000
        #  00000000000000000
        if n1 == 0:
            n1 = 1

        # FINDING ALL REC NODES
        for j in range(h):
            for i in range(t):
                nodesList.append((n1+i)+(j*nl))

        nodesList.sort()

        # MODIFYING CONTROL FUNCTION
        for node in nodesList:
            control[node-1,0] = value



