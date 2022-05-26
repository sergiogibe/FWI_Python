import numpy as np
import time
from matplotlib import pyplot as plt
from plot import *
from materialModel import *

'''======================== PARAMETERS ================================'''
length = 2                          # Total lenght of the domain [km].
depth  = 2                          # Total depth of the domain [km].
el = 100                            # Number of elements lenght.
ed = 100                            # Number of elements depth.
I = 1                               # Pulse intensity.
freq = 2                            # Pulse frequency [hz].
T = 2.6                             # Time of observation [s].
dt = 0.002                          # Newmark time delta [s].
delta=0.03                          # Reaction-diffusion pseudo-time.
niter=14                            # Total number of problem iterations.
diff=5*pow(10,-6)                   # Reaction-diffusion coefficient.
upper_vel = 3.5                     # Highest medium velocity [km/s].
lower_vel = 1.5                     # Lowest medium velocity [km/s].
normf = 0.1                         # Sensitivity normalizing factor.
regf  = 0.0008                      # Sensitivity regularization factor.
'''===================================================================='''

print("Example 1 - EXPERIMENTAL PROBLEM (SINGLE CENTERED SQUARE - LS).")

from levelSet import LevelSet
model = LevelSet(el=100,ed=100,
                 velocities=[1.5,3.5],
                 diff=5*pow(10,-6),
                 delta=0.03
                 )

from problem import Problem
realProblem = Problem(el=100,ed=100,length=2.0,depth=2.0,
                      I=1.0,freq=2.0,T=2.6,dt=0.002,
                      sourcesPos=[(0.60,0.60)],receiversPos=[(0.00,0.00)],
                      materialModel=model,
                      ABC=(0,2)
                      )

realProblem.solve(diag_scale=True,render=True,saveReceivers=False)

print(" ")
print("-"*40)
print("THANK YOU.")
print("Visit: https://github.com/sergiogibe ")
print("-"*40)
