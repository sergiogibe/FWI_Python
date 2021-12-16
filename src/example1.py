print("Example 1 - Synthetic data generating.")

''' Simple example of generating synthetic data. '''
#   PROBLEM: real_model.npy in data_dir (stratified layers)
#   Only works for FWI material model, to change that one needs
#   to re-work the values in the numpy array.

'''======================== PARAMETERS ================================'''
lenght = 5                          # Total lenght of the domain [km].
depth  = 3                          # Total depth of the domain [km].
el = 100                            # Number of elements lenght.
ed = 60                             # Number of elements depth.
I = 1                               # Pulse intensity.
freq = 2                            # Pulse frequency [hz].
T = 4.0                             # Time of observation [s].
dt = 0.002                          # Newmark time delta [s].
dispF = 1.0                         # Ricker pulse's displacement.
delta=0.03                          # Reaction-diffusion pseudo-time.
niter=100                           # Total number of problem iterations.
diff=5*pow(10,-6)                   # Reaction-diffusion coefficient.
upper_vel = 3.5                     # Highest medium velocity [km/s].
lower_vel = 1.5                     # Lowest medium velocity [km/s].
'''---------------------------------------------------------------------'''
env_config = [el,ed,lenght,depth]
pulse_config = [I,freq,T,dt,dispF]
opt_config = [delta,niter,diff]
vel_config = [upper_vel,lower_vel]
'''---------------------------------------------------------------------'''

import numpy as np
import time
from mesh import LinearMesh2D
mesh = LinearMesh2D(env_config,vel_config)


from source import Source
from receiver import Receiver
sources, receivers = [] , []

source = Source(0,0.3,2.7,mesh)
sources.append(source.nodalAbs)
sources = np.asarray(sources,dtype=np.int32)

for i in range(15):
    receiver = Receiver(0,0.3+(i*0.3),2.5,mesh)
    receivers.append(receiver.nodalAbs)
receivers = np.asarray(receivers,dtype=np.int32)


from rickerPulse import RickerPulse
pulse = RickerPulse(pulse_config)


from externalForce import ExternalForce
force = ExternalForce(sources,mesh.nNodes,pulse.pulse)


from materialModel import MatmodelFWI
matmod = MatmodelFWI(mesh,opt_config)
matmod.plot_design(sources,receivers)
matmod.read_RealModel()


from ElementFrame2D import ElementFrame
frame = ElementFrame(mesh,sparse_mode=True)


from mount import MountProblem
problem = MountProblem(mesh, matmod, frame, dataGen=True, diag_scale=True)


from timeSolver import solverEXP_CCompiled
print("Solving..")
start = time.time()
u = solverEXP_CCompiled(frame.stiff,problem.mass,force.force,pulse.deltaTime,sources)
end = time.time()
print(f"Elapsed time : {end - start} seconds            ")

from plot import render_propagating
render_propagating(mesh,u,size=10)



print(" ")
print("-"*100)
print("THANK YOU.")
print("Visit: https://github.com/sergiogibe/FWI_SV_2021 ")
print("-"*100)