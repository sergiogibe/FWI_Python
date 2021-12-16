import numpy as np

class ExternalForce:
    
    def __init__(self,sources,nNodes,pulse):
        
        self.sourcePosition = sources
        self.pulse          = pulse
        self.nSources       = sources.shape[0]
        self.nNodes         = nNodes
        self.steps          = pulse.shape[1]

        self.force = np.empty([self.nSources,self.steps],dtype=np.float32)
        self.pulse[0, 0] = 0.0
        
        for j in range(0,self.steps):
            for i in range(0,self.nSources):
                self.force[i,j] = self.pulse[0,j]

             