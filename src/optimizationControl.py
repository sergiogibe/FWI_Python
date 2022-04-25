from timeSolver          import solverF_CCompiled, solverS_CCompiled
from plot                import *
import time

def backTracking_LineS(mesh, stiff, frame, mass, force, dt, cost0, cF, matmod, sourceList, receiverList, data, it, niter,
                       sens, difA, difB, fig, startit, resetDelta, plotModel, plotSens):

    #MODIFIED ARMIJO-GOLDSTEIN ALGORITHM
    v = 0
    cost = 0
    misfit = 0
    deltaChange = matmod.delta
    control = 1
    controlMax = 10

    while control <= controlMax:
        print("-" * 50)
        print(f"Solving forward and cost funct..         ..   OK.")
        start = time.time()
        v, cost, misfit = solverF_CCompiled(stiff, mass, force, dt, sourceList, receiverList, data)
        end = time.time()
        if it >= niter:
            break
        print(f"Elapsed time : {end - start} seconds            ")
        print("-" * 50)
        print(f"Cost functional:        {cost}                  ")
        print("-" * 50)
        if cost < cost0:
            matmod.delta *= cost/cost0
            matmod.update(sens, difA, difB)
            matmod.limit()
            matmod.writeHist()
            cF.append(cost)
            plt.close(fig)
            fig = plot_cost(cF, it, niter)
            cost0 = cost
            if plotModel:
                matmod.plot_model(ID=it + 1)
            if plotSens:
                plot_field(mesh, sens, ncol=0, field_name="sensitivity", ID=it + 1)
            it+=1
            print("-" * 50)
            endit = time.time()
            print(f"TOTAL ITERATION TIME : {endit - startit} seconds")
            print("-=" * 25)
            print(f"Iteration {it + 1} - Solving..                  ")
            print("-" * 50)
            startit = time.time()
            print("Mounting new problem..                   ..   OK.")
            mass = matmod.mount_problem(frame, diag_scale=True, dataGen=False)
        else:
            print("-" * 50)
            print(f"Backtracking {control}..                        ..      ")
            matmod.delta *= 0.5
            matmod.update(sens, difA, difB)
            matmod.limit()
            print("-" * 50)
            print("Mounting new problem..                   ..   OK.")
            mass = matmod.mount_problem(frame, diag_scale=True, dataGen=False)
            control+=1
        if control == controlMax + 1:
            print("-" * 50)
            print(f"WARNING: Tried for {controlMax} times   ..      ")
            print("Needs new direction!                     ..      ")
            it+=1
            endit = time.time()
            print("-" * 50)
            print(f"TOTAL ITERATION TIME : {endit - startit} seconds")

    if resetDelta:
        matmod.delta = deltaChange
    matmod.writeHist()
    if plotModel:
        matmod.plot_model(ID=it + 1)  # TODO LS - APPLY PROPERTY is SLOW!
    if plotSens:
        plot_field(mesh, sens, ncol=0, field_name="sensitivity", ID=it + 1)

    return v, cost, misfit, it, fig

