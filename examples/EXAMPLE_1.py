import FWI_Python as fwi
import os

os.system("clear")
print("\nExample 1 - EXPERIMENTAL PROBLEM (SINGLE SQUARE - LS).\n")

print("Creating model")
realModel = fwi.LevelSet(el=100,ed=100,
                         length=2.00, depth=2.00,
                         velocities=[1.0,3.0]
                         )
sources   = [(0.20,0.20),(1.00,0.20),(1.80,0.20)]
receivers = [(0.20+0.07*i,0.30) for i in range(24)]

print("Creating problem (EXP)")
realProblem = fwi.Problem(el=100,ed=100,length=2.0,depth=2.0,
                          I=1.0,freq=2.0,T=2.6,dt=0.002,
                          sourcesPos=sources,receiversPos=receivers,
                          materialModel=realModel,
                          ABC=(0,2),
                          saveResponse=True
                          )
fwi.make_inclusions([(1.2,1.6,0.8,1.0)],realProblem.mesh,realModel.control,value=0.01)
realModel.plot("plotDesign_TEST", plotSR=[sources,receivers])
realProblem.solve()

print("\nSetting up for inversion.. ")
rd = fwi.RD(tau=5*pow(10,-6),
            delta=0.5,
            frame=realProblem.frame
            )

model = fwi.LevelSet(el=100,ed=100,
                     length=2.00, depth=2.00,
                     velocities=[1.0,3.0]
                     )

invProblem = fwi.Problem(el=100,ed=100,length=2.0,depth=2.0,
                         I=1.0,freq=2.0,T=2.6,dt=0.002,
                         sourcesPos=sources,receiversPos=receivers,
                         materialModel=model,
                         ABC=(0,2)
                         )

fobj = []
for it in range(13):
    print(f"\nSolving iteration: {it+1}")
    invProblem.solve(exp=realProblem.exp)
    fobj.append(invProblem.obj)

    model.update(sens=invProblem.sens,
                 kappa=0.0008, c=1.0,
                 reacDiff=rd,
                 nametag=f"TEST_it_{it+1}",
                 savePlot=False
                 )

model.plot("FinalInv",save=True,plotSR=[sources,receivers])
fwi.plot_f(fobj)
