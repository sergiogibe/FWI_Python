import numpy as np
import time
from matplotlib import pyplot as plt
from plot import *
from materialModel import *
from levelSet import LevelSet
from problem import Problem
from reactionDiffusion import RD

'''=========================== PARAMETERS ================================'''
''' length                       # Total lenght of the domain [km].
    depth                        # Total depth of the domain [km].
    el                           # Number of elements lenght.
    ed                           # Number of elements depth.
    I                            # Pulse intensity.
    freq                         # Pulse frequency [hz].
    T                            # Time of observation [s].
    dt                           # Newmark time delta [s].
    delta                        # Reaction-diffusion pseudo-time.
    niter                        # Total number of problem iterations.
    diff                         # Reaction-diffusion coefficient.
    velocities                   # Medium velocity [km/s].
    normf                        # Sensitivity normalizing factor.
    regf                         # Sensitivity regularization factor.     '''
'''======================================================================='''

print("\nExample 1 - EXPERIMENTAL PROBLEM (SINGLE CENTERED SQUARE - LS).\n")

print("Creating model")
model = LevelSet(el=100,ed=100,
                 length=2.00, depth=2.00,
                 velocities=[1.5,3.5]
                 )

print("Creating problem (EXP)")
realProblem = Problem(el=100,ed=100,length=2.0,depth=2.0,
                      I=1.0,freq=2.0,T=2.6,dt=0.002,
                      sourcesPos=[(0.60,0.60)],receiversPos=[(0.00,0.00)],
                      materialModel=model,
                      ABC=(0,2),
                      saveResponse=True
                      )
print("\nSolving")
realProblem.solve()

print("\nSetting up Reaction Diffusion")
rd = RD(tau=5*pow(10,-6),
        delta=0.03,
        frame=realProblem.frame
        )

print("\nCreating problem (INV)")
invProblem = Problem(el=100,ed=100,length=2.0,depth=2.0,
                     I=1.0,freq=2.0,T=2.6,dt=0.002,
                     sourcesPos=[(0.60,0.60)],receiversPos=[(0.00,0.00)],
                     materialModel=model,
                     ABC=(0,2)
                     )
print("\nSolving")
invProblem.solve(exp=realProblem.exp)

print("\nUpdating model")
model.update(sens=invProblem.sens,
             kappa=0.0008, c=0.1,
             reacDiff=rd,
             nametag="FirstUpdate_TEST"
             )



print(" ")
print("-"*40)
print("Visit: https://github.com/sergiogibe ")
print("-"*40)
