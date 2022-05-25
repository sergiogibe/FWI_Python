print("Example 3 - Simple square inversion using LS.")
import numpy as np
import time
from matplotlib import pyplot as plt
from plot import *

''' Example of inverting a simple square. '''
#   PROBLEM: Simple centered square.

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
niter=14                            # Total number of problem iterations.
diff=5*pow(10,-6)                   # Reaction-diffusion coefficient.
upper_vel = 3.5                     # Highest medium velocity [km/s].
lower_vel = 1.5                     # Lowest medium velocity [km/s].
normf = 0.1                         # Sensitivity normalizing factor.
regf  = 0.0008                      # Sensitivity regularization factor.
'''---------------------------------------------------------------------'''
env_config = [el,ed,lenght,depth]
pulse_config = [I,freq,T,dt,dispF]
opt_config = [delta,niter,diff]
'''---------------------------------------------------------------------'''

from mesh import LinearMesh2D
mesh = LinearMesh2D(env_config)

from pml import PML
pmlObj = PML(mesh,thickness=20,atten=2)
pmlObj.plot_PML()

'''---------------------------------------------------------------------'''

from source import Source
sources = []
# source1 = Source(0,0.25,0.25,mesh)
# source2 = Source(0,0.25,1.75,mesh)
# source3 = Source(0,1.75,0.25,mesh)
# source4 = Source(0,1.75,1.75,mesh)
# sources.append(source1.nodalAbs)
# sources.append(source2.nodalAbs)
# sources.append(source3.nodalAbs)
# sources.append(source4.nodalAbs)
source1 = Source(0,1.00,1.00,mesh)
sources.append(source1.nodalAbs)
#sources.sort()
sources = np.asarray(sources,dtype=np.int32) #Nodal positions (starts with 1 !!)

#plot_inDomain(sources,mesh,"sources_check","0")

'''---------------------------------------------------------------------'''

from receiver import Receiver
receivers = []
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
receivers = np.asarray(receivers,dtype=np.int32) #Nodal positions (starts with 1 !!)

'''---------------------------------------------------------------------'''

from rickerPulse import RickerPulse
pulse = RickerPulse(pulse_config)

'''-------TOGETHER-----------------------------------'''

from externalForce import ExternalForce
force = ExternalForce(sources,mesh.nNodes,pulse.pulse)

'''---------------------------------------------------------------------'''

from materialModel import *
control = MatmodelLS(mesh, opt_config)
control.set_velocities([lower_vel,upper_vel])
control.square_dist()
#control.plot_design(sources, receivers)

colors = [
    (0.5, 0.0, 0.0),  # 'red',
    (1.0, 0.6, 0.0),  # 'orange',
    (0.0, 0.5, 1.0),  # 'cyan',
    (0.0, 0.0, 0.5),  # 'blue',
    (0.0, 0.0, 0.0)  # 'black'
]
vp = np.zeros((101,101))
for j in range(101):
    for i in range(101):
        vp[j,i]=control.realModel[i+j*101]
plot_contour(2,vp,(-1,1),levels=[-1.0,0.0,1.0],colors=colors,fill=True)

'''---------------------------------------------------------------------'''

from ElementFrame2D import ElementFrame
frame = ElementFrame(mesh,sparse_mode=True)

'''---------------------------------------------------------------------'''

from timeSolver import *
print("Generating synth data..")
exp_problem = control.mount_problem(frame, diag_scale=True, dataGen=True)
u = np.zeros([mesh.nNodes,pulse.steps,sources.shape[0]],dtype=np.float32)
for sh in range(0,sources.shape[0]):
    u[:,:,sh] = solverEXP1shotPML_CCompiled(frame.stiff,exp_problem,force.force,pulse.deltaTime,
                                         sources[sh],sh,pmlObj).base

exp = np.float32(np.zeros([receivers.shape[0],pulse.steps,sources.shape[0]]))
for sh in range(0,sources.shape[0]):
    for rc in range(0,receivers.shape[0]):
        exp[rc,:,sh] = u[receivers[rc]-1,:,sh]

render_propagating(mesh,u[:,:,0],size=10)

'''---------------------------------------------------------------------'''

# #INITIAL SETTINGS
# print('Prepering for inversion.. ')
# control.homoGuess()
# costFunction, costFunctionOld = 0, 1
# minSequence = []
# sensitivity = np.zeros([mesh.nNodes,1],dtype=np.float32)
# regA, regB = frame.getConsistent(tau=1)
# difA, difB = frame.getConsistent(tau=diff)
# print('-'*30)
#
# # FOR LOOP GLOBAL ITER
# for it in range(0,niter):
#     print(f'Iteration {it+1}.. ')
#
#     # MOUNT THE PROBLEM
#     print('MOUNT THE PROBLEM')
#     problem = control.mount_problem(frame, diag_scale=True, dataGen=False)
#
#     # FOR LOOP SHOTS
#     for shot in range(0,sources.shape[0]):
#         # FORWARD PROBLEM
#         print(f'FORWARD PROBLEM - SHOT {shot+1}')
#         v, cost, misfit = solverF1shot_CCompiled(frame.stiff, problem, force.force, dt,
#                                                   sources[shot], shot, receivers, exp)
#         # ADJOINT PROBLEM
#         print(f'ADJOINT PROBLEM - SHOT {shot+1}')
#         sens = solverS_CCompiled(frame.stiff, problem, misfit, dt, receivers, v)
#
#         costFunction += cost
#         sensitivity += sens
#
#     # SUM SHOT SOLUTIONS
#     print('MODIFY SENSITIVITY')
#     sensitivity = control.modSens(sensitivity, regf, normf, regA, regB, regSens=True, normSens=True)
#
#     # UPDATE THE MODEL AND PLOT
#     print('UPDATE MODEL AND PLOT')
#     control.control_step(costFunction,costFunctionOld,it,boost=1.1,manualReset=False)
#     control.update_reactdiff(sensitivity,difA,difB)
#     control.plot_model(ID=it+1)
#     control.writeHist()
#
#     # LINE SEARCH STEP CONTROL
#     print(f'COST FUNCTION = {costFunction}')
#     minSequence.append(costFunction)
#
#     # RESET INTERNAL VARIABLES
#     costFunctionOld = costFunction
#     costFunction = 0
#     sensitivity = np.zeros([mesh.nNodes,1],dtype=np.float32)
#     print('-'*30)
#
# plot_cost(minSequence,niter,niter)

'''---------------------------------------------------------------------'''

print(" ")
print("-"*40)
print("THANK YOU.")
print("Visit: https://github.com/sergiogibe ")
print("-"*40)
