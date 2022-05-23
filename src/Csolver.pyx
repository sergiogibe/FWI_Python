import numpy as np
cimport cython
from cython.parallel import prange

DTYPE = np.float32

@cython.boundscheck(False)
@cython.wraparound(False)
def field_updateEXP(float[:,::1] ut0, float[:,::1] vt0, float[:,::1] at0, float[:,::1] at1, float[:,::1] field, int tt, float dt):

    cdef Py_ssize_t dof = field.shape[0]
    cdef Py_ssize_t steps = field.shape[1]
    cdef Py_ssize_t n
    cdef Py_ssize_t t = tt

    result1 = np.zeros((dof,1),dtype=DTYPE)
    result2 = np.zeros((dof,1),dtype=DTYPE)
    result3 = np.zeros((dof,1),dtype=DTYPE)
    cdef float[:,::1] result_view1 = result1
    cdef float[:,::1] result_view2 = result2
    cdef float[:,::1] result_view3 = result3

    for n in prange(dof,nogil=True):
        result_view1[n,0] = ut0[n,0] + dt*vt0[n,0] + dt*dt*0.5*at0[n,0]
        result_view2[n,0] = vt0[n,0] + dt*0.5*(at0[n,0] + at1[n,0])
        result_view3[n,0] = at1[n,0]

        field[n,t] = result_view1[n,0]

    return result1, result2, result3, field


@cython.boundscheck(False)
@cython.wraparound(False)
def field_updateEXPPML(float[:,::1] ut0, float[:,::1] vt0, float[:,::1] at0, float[:,::1] at1, float[:,::1] field, int tt, float dt, float[::1] forcePML):

    cdef Py_ssize_t dof = field.shape[0]
    cdef Py_ssize_t steps = field.shape[1]
    cdef Py_ssize_t n
    cdef Py_ssize_t l
    cdef Py_ssize_t t = tt

    result1 = np.zeros((dof,1),dtype=DTYPE)
    result2 = np.zeros((dof,1),dtype=DTYPE)
    result3 = np.zeros((dof,1),dtype=DTYPE)
    cdef float[:,::1] result_view1 = result1
    cdef float[:,::1] result_view2 = result2
    cdef float[:,::1] result_view3 = result3

    for n in prange(dof,nogil=True):
        at1[n,0] *= forcePML[n]
        result_view1[n,0] = ut0[n,0] + dt*vt0[n,0] + dt*dt*0.5*at0[n,0]
        result_view2[n,0] = vt0[n,0] + dt*0.5*(at0[n,0] + at1[n,0])
        result_view3[n,0] = at1[n,0]

        field[n,t] = result_view1[n,0]

    return result1, result2, result3, field

@cython.boundscheck(False)
@cython.wraparound(False)
def field_updateF(float[:,::1] ut0, float[:,::1] vt0, float[:,::1] at0, float[:,::1] at1, float[:,::1] field, int tt, float dt):

    cdef Py_ssize_t dof = field.shape[0]
    cdef Py_ssize_t steps = field.shape[1]
    cdef Py_ssize_t n
    cdef Py_ssize_t t = tt

    result1 = np.zeros((dof,1),dtype=DTYPE)
    result2 = np.zeros((dof,1),dtype=DTYPE)
    result3 = np.zeros((dof,1),dtype=DTYPE)
    cdef float[:,::1] result_view1 = result1
    cdef float[:,::1] result_view2 = result2
    cdef float[:,::1] result_view3 = result3

    for n in prange(dof,nogil=True):
        result_view1[n,0] = ut0[n,0] + dt*vt0[n,0] + dt*dt*0.5*at0[n,0]
        result_view2[n,0] = vt0[n,0] + dt*0.5*(at0[n,0] + at1[n,0])
        result_view3[n,0] = at1[n,0]

        field[n,t] = result_view2[n,0]

    return result1, result2, result3, field


@cython.boundscheck(False)
@cython.wraparound(False)
def field_updateS(float[:,::1] ut0, float[:,::1] vt0, float[:,::1] at0, float[:,::1] at1, int tt, float dt):

    cdef Py_ssize_t dof = ut0.shape[0]
    cdef Py_ssize_t n
    cdef Py_ssize_t t = tt

    result1 = np.zeros((dof,1),dtype=DTYPE)
    result2 = np.zeros((dof,1),dtype=DTYPE)
    result3 = np.zeros((dof,1),dtype=DTYPE)
    cdef float[:,::1] result_view1 = result1
    cdef float[:,::1] result_view2 = result2
    cdef float[:,::1] result_view3 = result3

    for n in prange(dof,nogil=True):
        result_view1[n,0] = ut0[n,0] + dt*vt0[n,0] + dt*dt*0.5*at0[n,0]
        result_view2[n,0] = vt0[n,0] + dt*0.5*(at0[n,0] + at1[n,0])
        result_view3[n,0] = at1[n,0]

    return result1, result2, result3


@cython.boundscheck(False)
@cython.wraparound(False)
def marchEXP(float[:,:]f, float[:,::1]M, float[:,:]barK, int tt, int[::1] sourceList):

    cdef Py_ssize_t dof = M.shape[0]
    cdef Py_ssize_t nshot = sourceList.shape[0]
    cdef Py_ssize_t n
    cdef Py_ssize_t l
    cdef Py_ssize_t t = tt
    cdef float frc

    result = np.zeros((dof,1),dtype=DTYPE)
    cdef float[:,::1] result_view = result

    for n in prange(dof,nogil=True):

        frc = 0.0
        for l in range(nshot):
            if n+1 == sourceList[l]:
                frc = f[l, t]

        result_view[n,0] = (frc - barK[n,0]) / M[n,0]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def marchEXP1shot(float[:,:]f, float[:,::1]M, float[:,:]barK, int tt, int shotNode, int shot):

    cdef Py_ssize_t dof = M.shape[0]
    cdef Py_ssize_t n
    cdef Py_ssize_t l
    cdef Py_ssize_t t = tt
    cdef Py_ssize_t sN = shotNode
    cdef Py_ssize_t s = shot
    cdef float frc

    result = np.zeros((dof,1),dtype=DTYPE)
    cdef float[:,::1] result_view = result

    for n in prange(dof,nogil=True):

        frc = 0.0

        if n+1 == sN:
            frc = f[s, t]

        result_view[n,0] = (frc - barK[n,0]) / M[n,0]

    return result



@cython.boundscheck(False)
@cython.wraparound(False)
def marchF(float[:,:]f, float[:,::1]M, float[:,:]barK, float[:,::1]ut0, float[:,::1]data, float[:,::1]misfit, int tt, float cost, int[::1]sourceList, int[::1]receiverList):

    cdef Py_ssize_t dof = M.shape[0]
    cdef Py_ssize_t nshot = sourceList.shape[0]
    cdef Py_ssize_t nrec = receiverList.shape[0]
    cdef Py_ssize_t n
    cdef Py_ssize_t l
    cdef Py_ssize_t r
    cdef Py_ssize_t t = tt
    cdef float frc
    cdef float aux = 0

    result = np.zeros((dof,1),dtype=DTYPE)
    cdef float[:,::1] result_view = result

    for n in prange(dof,nogil=True):

        frc = 0.0
        for l in range(nshot):
            if n+1 == sourceList[l]:
                frc = f[l, t]

        result_view[n,0] = (frc - barK[n,0]) / M[n,0]

        for r in range(nrec):
            if n+1 == receiverList[r]:
                aux = ut0[n,0]-data[r,t-1]
                misfit[r,t] = aux
                cost += 0.5*aux*aux

    return result, misfit, cost



@cython.boundscheck(False)
@cython.wraparound(False)
def marchF1shot(float[:,:]f, float[:,::1]M, float[:,:]barK, float[:,::1]ut0, float[:,::1]data, float[:,::1]misfit, int tt, float cost, int shotNode, int shot, int[::1]receiverList):

    cdef Py_ssize_t dof = M.shape[0]
    cdef Py_ssize_t sN = shotNode
    cdef Py_ssize_t s = shot
    cdef Py_ssize_t nrec = receiverList.shape[0]
    cdef Py_ssize_t n
    cdef Py_ssize_t l
    cdef Py_ssize_t r
    cdef Py_ssize_t t = tt
    cdef float frc
    cdef float aux = 0

    result = np.zeros((dof,1),dtype=DTYPE)
    cdef float[:,::1] result_view = result

    for n in prange(dof,nogil=True):

        frc = 0.0
        if n+1 == sN:
            frc = f[s, t]

        result_view[n,0] = (frc - barK[n,0]) / M[n,0]

        for r in range(nrec):
            if n+1 == receiverList[r]:
                aux = ut0[n,0]-data[r,t-1]
                misfit[r,t] = aux
                cost += 0.5*aux*aux

    return result, misfit, cost



@cython.boundscheck(False)
@cython.wraparound(False)
def marchS(float[:,:]f, float[:,::1]M, float[:,:]barK, float[:,::1]vt0, float[:,::1]v, float[:,::1]gt1, float[:,::1]gt2, float[:,::1]sens, int tt, int stps, float dt, int[::1] receiverList):

    cdef Py_ssize_t dof = M.shape[0]
    cdef Py_ssize_t nrec = receiverList.shape[0]
    cdef Py_ssize_t n
    cdef Py_ssize_t r
    cdef Py_ssize_t steps = stps
    cdef Py_ssize_t t = tt
    cdef float frc

    result = np.zeros((dof,1),dtype=DTYPE)
    cdef float[:,::1] result_view = result

    for n in prange(dof,nogil=True):

        frc = 0.0
        for r in range(nrec):
            if n+1 == receiverList[r]:
                frc = f[r,t]

        result_view[n,0] = (frc - barK[n,0]) / M[n,0]

        #(correction for vt0 = 0 in t = 1; so t has to start in 3)
        if t % 2 != 0 and t > 2:
            gt1[n, 0] = v[n,steps-t]*vt0[n,0]
        if t % 2 == 0 and t > 2:
            gt2[n,0] = v[n,steps-t]*vt0[n,0]
            sens[n,0] += 0.5*dt*gt1[n,0] + 0.5*dt*gt2[n,0]

    return result
