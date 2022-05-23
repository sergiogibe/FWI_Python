import numpy as np
import Csolver

def solverEXP_CCompiled(K,M,f,dt,sourceList):

    #Useful parameters
    steps = f.shape[1]
    dof   = M.shape[0]

    #Fields saved
    u   = np.zeros([dof, steps],dtype=np.float32)

    #Fields not saved
    at0 = np.zeros([dof, 1],dtype=np.float32)
    ut0 = np.zeros([dof, 1],dtype=np.float32)
    vt0 = np.zeros([dof, 1],dtype=np.float32)

    #Turn contiguous (precaution)
    np.ascontiguousarray(M)

    #Solution
    for t in range(1, steps):

        barK = K.dot((ut0 + dt * vt0 + 0.5 * dt * dt * at0))

        at1 = Csolver.marchEXP(f, M, barK,t, sourceList)

        #Updating and saving field:
        ut0, vt0, at0, u = Csolver.field_updateEXP(ut0, vt0, at0, at1, u, t, dt)

    return u


def solverEXP1shot_CCompiled(K,M,f,dt,shotNode,shot):

    #Useful parameters
    steps = f.shape[1]
    dof   = M.shape[0]

    #Fields saved
    u   = np.zeros([dof, steps],dtype=np.float32)

    #Fields not saved
    at0 = np.zeros([dof, 1],dtype=np.float32)
    ut0 = np.zeros([dof, 1],dtype=np.float32)
    vt0 = np.zeros([dof, 1],dtype=np.float32)

    #Turn contiguous (precaution)
    np.ascontiguousarray(M)

    #Solution
    for t in range(1, steps):

        barK = K.dot((ut0 + dt * vt0 + 0.5 * dt * dt * at0))

        at1 = Csolver.marchEXP1shot(f, M, barK,t, shotNode, shot)

        #Updating and saving field:
        ut0, vt0, at0, u = Csolver.field_updateEXP(ut0, vt0, at0, at1, u, t, dt)

    return u


def solverEXP1shotPML_CCompiled(K,M,f,dt,shotNode,shot,pmlObj):

    #Useful parameters
    steps = f.shape[1]
    dof   = M.shape[0]

    #Fields saved
    u   = np.zeros([dof, steps],dtype=np.float32)

    #Fields not saved
    at0 = np.zeros([dof, 1],dtype=np.float32)
    ut0 = np.zeros([dof, 1],dtype=np.float32)
    vt0 = np.zeros([dof, 1],dtype=np.float32)

    #Turn contiguous (precaution)
    np.ascontiguousarray(M)

    #Solution
    for t in range(1, steps):

        barK = K.dot((ut0 + dt * vt0 + 0.5 * dt * dt * at0))

        at1 = Csolver.marchEXP1shot(f, M, barK,t, shotNode, shot)

        #Updating and saving field:
        ut0, vt0, at0, u = Csolver.field_updateEXPPML(ut0, vt0, at0, at1, u, t, dt, pmlObj.forcePML)

    return u


def solverF_CCompiled(K,M,f,dt,sourceList,receiverList,data):

    #Useful parameters
    steps = f.shape[1]
    dof   = M.shape[0]
    cost = 0

    #Fields saved
    v = np.ascontiguousarray(np.zeros([dof, steps],dtype=np.float32))
    misfit = np.ascontiguousarray(np.zeros([len(receiverList),steps],dtype=np.float32))

    #Fields not saved
    at0 = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))
    ut0 = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))
    vt0 = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))

    #Turn contiguous
    np.ascontiguousarray(M)

    #Solution
    for t in range(1, steps):

        barK = K.dot((ut0 + dt * vt0 + 0.5 * dt * dt * at0))

        at1, misfit, cost = Csolver.marchF(f, M, barK, ut0, data, misfit, t, cost, sourceList, receiverList)

        #Updating and saving field:
        ut0, vt0, at0, v = Csolver.field_updateF(ut0, vt0, at0, at1, v, t, dt)

    return v, cost, misfit



def solverF1shot_CCompiled(K,M,f,dt,shotNode,shot,receiverList,data):

    #Useful parameters
    steps = f.shape[1]
    dof   = M.shape[0]
    cost = 0

    #Fields saved
    v = np.ascontiguousarray(np.zeros([dof, steps],dtype=np.float32))
    misfit = np.ascontiguousarray(np.zeros([len(receiverList),steps],dtype=np.float32))

    #Fields not saved
    at0 = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))
    ut0 = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))
    vt0 = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))

    #Turn contiguous
    np.ascontiguousarray(M)

    #Solution
    for t in range(1, steps):

        barK = K.dot((ut0 + dt * vt0 + 0.5 * dt * dt * at0))

        at1, misfit, cost = Csolver.marchF1shot(f, M, barK, ut0, np.ascontiguousarray(data[:,:,shot]), misfit, t,
                                                cost, shotNode, shot, receiverList)

        #Updating and saving field:
        ut0, vt0, at0, v = Csolver.field_updateF(ut0, vt0, at0, at1, v, t, dt)

    return v, cost, misfit


def solverS_CCompiled(K,M,misfit,dt,receiverList,v):

    #Useful parameters
    steps = misfit.shape[1]
    dof   = M.shape[0]

    #Flip misfit
    misfit = np.ascontiguousarray(np.fliplr(misfit))

    #Fields saved
    sens = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))

    # Turn contiguous
    np.ascontiguousarray(M)
    np.ascontiguousarray(v)

    #Init accel (final accel bc backwards)
    at0 = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))
    for n in range(0,dof):
        for r in range(0,receiverList.shape[0]):
            if n+1 == receiverList[r]:
                at0[n,0] = misfit[r,0] / M[n,0]

    #Fields not saved
    at1 = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))
    ut0 = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))
    vt0 = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))

    gt1 = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))
    gt2 = np.ascontiguousarray(np.zeros([dof, 1],dtype=np.float32))

    #Solution
    for t in range(1, steps):

        barK = K.dot((ut0 + dt * vt0 + 0.5 * dt * dt * at0))

        at1 = Csolver.marchS(misfit, M, barK, vt0, v, gt1, gt2, sens, t, steps, dt, receiverList)

        #Updating:
        ut0, vt0, at0 = Csolver.field_updateS(ut0, vt0, at0, at1, t, dt)

    return sens