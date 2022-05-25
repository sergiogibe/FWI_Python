import numpy as np

class Problem:
    def __init__(self,
                 el: int, ed: int, lengh: float, depth: float, ABCt: int,
                 I: float, freq: float, T: float, dt: float,
                 receivers: object,
                 sources: object,
                 ABC = False,
                 ):
        # INSTANCE ATRIBUTTES


        # GENERATE MESH
        from mesh import LinearMesh2D
        mesh: object = LinearMesh2D([el,ed,lenght,depth])

        # APPLY ABSORBING CONDITIONS
        if ABC == True:
            from pml import PML
            absLayer: object = PML(mesh, thickness=ABCt, atten=2)
            # deve entrar isso no construtor: pmlObj.plot_PML()

        # SET EXTERNAL FORCES
        from rickerPulse import RickerPulse
        pulse: object = RickerPulse([I,freq,T,dt,1.0])
        from externalForce import ExternalForce
        force: object = ExternalForce(sources, mesh.nNodes, pulse.pulse)



def mount():
    from ElementFrame2D import ElementFrame
    frame = ElementFrame(mesh, sparse_mode=True)

def solve():
    pass

