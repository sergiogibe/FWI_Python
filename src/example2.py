print('Deprecated example')
exit()
print("Example 1 - Simple square inversion.")
import numpy as np
import time
from matplotlib import pyplot as plt

''' Example of inverting a simple square. '''
#   PROBLEM: Simple centered square real distribution.
#   It works for FWI, SIMP and LS material models.
#   It doesn't work for MMLS material model.

'''======================== PARAMETERS ================================'''
lenght = 2                          # Total lenght of the domain [km].
depth  = 2                          # Total depth of the domain [km].
el = 100                            # Number of elements lenght.
ed = 100                            # Number of elements depth.
I = 1                               # Pulse intensity.
freq = 2                            # Pulse frequency [hz].
T = 2.6                             # Time of observation [s].
dt = 0.002                          # Newmark time delta [s].
dispF = 1.0                         # Ricker pulse's displacement.
delta=0.03                          # Reaction-diffusion pseudo-time.
niter=50                            # Total number of problem iterations.
diff=5*pow(10,-6)                   # Reaction-diffusion coefficient.
upper_vel = 3.5                     # Highest medium velocity [km/s].
lower_vel = 1.5                     # Lowest medium velocity [km/s].
normf = 0.1                         # Sensitivity normalizing factor.
regf  = 0.0008                      # Sensitivity regularization factor.
'''---------------------------------------------------------------------'''
env_config = [el,ed,lenght,depth]
pulse_config = [I,freq,T,dt,dispF]
opt_config = [delta,niter,diff]
vel_config = [upper_vel,lower_vel]
'''---------------------------------------------------------------------'''

from mesh import LinearMesh2D
mesh = LinearMesh2D(env_config,vel_config)


from source import Source
from receiver import Receiver
sources, receivers = [] , []

source1 = Source(0,0.05,0.05,mesh)
source2 = Source(0,0.05,1.95,mesh)
source3 = Source(0,1.95,0.05,mesh)
source4 = Source(0,1.95,1.95,mesh)
sources.append(source1.nodalAbs)
sources.append(source2.nodalAbs)
sources.append(source3.nodalAbs)
sources.append(source4.nodalAbs)
sources.sort()
sources = np.asarray(sources,dtype=np.int32)

for i in range(15):
    receiver1 = Receiver(0, 0.05, 0.05 + (i * 0.13), mesh)
    receiver2 = Receiver(0, 1.95, 0.05 + (i * 0.13), mesh)
    receiver3 = Receiver(0, 0.05 + (i * 0.13), 0.05, mesh)
    receiver4 = Receiver(0, 0.05 + (i * 0.13), 1.95, mesh)
    receivers.append(receiver1.nodalAbs)
    receivers.append(receiver2.nodalAbs)
    receivers.append(receiver3.nodalAbs)
    receivers.append(receiver4.nodalAbs)
receivers = list(dict.fromkeys(receivers))
receivers.sort()
receivers = np.asarray(receivers,dtype=np.int32)


from rickerPulse import RickerPulse
pulse = RickerPulse(pulse_config)


from externalForce import ExternalForce
force = ExternalForce(sources,mesh.nNodes,pulse.pulse)


from materialModel import *
matmod = MatmodelLS(mesh,opt_config)
matmod.plot_design(sources,receivers)
matmod.square_dist()


from ElementFrame2D import ElementFrame
frame = ElementFrame(mesh,sparse_mode=True)


from mount import MountProblem
problem = MountProblem(mesh, matmod, frame, dataGen=True, diag_scale=True)


from timeSolver import *
print("Solving..")
start = time.time()
u = solverEXP_CCompiled(frame.stiff,problem.mass,force.force,pulse.deltaTime,sources)
end = time.time()
print(f"Elapsed time : {end - start} seconds            ")

exp = np.float32(np.zeros([receivers.shape[0],pulse.steps]))
for n in range(0,mesh.nNodes):
    if n+1 in receivers:
        exp[np.where(receivers==n+1),:] = u[n,:]


from plot import *
render_propagating(mesh,u,size=10)

matmod.homoGuess()
matmod.plot_model(ID=0)
mass = matmod.mount_problem(frame, diag_scale=True, dataGen=False)

regA, regB = frame.getConsistent(tau=1)
difA, difB = frame.getConsistent(tau=diff)

from optimizationControl import backTracking_LineS
print("Starting iterations..                    ..      ")
start_inv = time.time()
sens = 0
cF   = []
cost0 = 0
misfit = 0
v = 0
it = 0
fig = plt.figure(figsize=(10, 5))
while it < niter:
    print("-=" * 25)
    print(f"Iteration {it + 1} - Solving..                  ")
    startit = time.time()
    print("-" * 50)
    print("Mounting problem..                       ..      ")
    start = time.time()
    mass = matmod.mount_problem(frame, diag_scale=True, dataGen=False)
    end = time.time()
    print("                                         ..   OK.")
    print(f"Elapsed time : {end - start} seconds            ")
    if it == 0:
        print("-"*50)
        print("Solving forward and cost funct..         ..      ")
        start = time.time()
        v, cost0, misfit = solverF_CCompiled(frame.stiff, mass, force.force, dt, sources, receivers, exp)
        end = time.time()
        cF.append(cost0)
        plt.close(fig)
        fig = plot_cost(cF, it, niter)
        print("                                         ..   OK.")
        print(f"Elapsed time : {end - start} seconds            ")
        print("-" * 50)
        print(f"Cost functional:        {cost0}                 ")
    print("-" * 50)
    print("Solving adjoint and sensitivity..        ..      ")
    start = time.time()
    sens = solverS_CCompiled(frame.stiff,mass,misfit,dt,receivers,v)
    end = time.time()
    print("                                         ..   OK.")
    print(f"Elapsed time : {end - start} seconds            ")
    print("-"*50)
    print("Modifying sens and updating dist..       ..      ")
    start = time.time()
    sens = matmod.modSens(sens,regf,normf,regA,regB,regSens=True,normSens=True)
    matmod.update(sens,difA,difB)
    matmod.limit()
    mass = matmod.mount_problem(frame, diag_scale=True, dataGen=False)
    end = time.time()
    print("                                         ..   OK.")
    print(f"Elapsed time : {end - start} seconds            ")
    print("-"*50)
    print("Optimization process..                   ..      ")
    v, cost, misfit, it, fig = backTracking_LineS(mesh, frame.stiff, frame, mass, force.force, pulse.deltaTime, cost0, cF, matmod,
                                                  sources, receivers, exp, it, niter, sens, difA, difB, fig, startit,
                                                  resetDelta=True, plotModel=True, plotSens=True)


end_inv = time.time()
print(f"Elapsed time : {end_inv - start_inv} seconds            ")



print(" ")
print("-"*100)
print("THANK YOU.")
print("Visit: https://github.com/sergiogibe/FWI_SV_2021 ")
print("-"*100)