import numpy as np

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