import numba as nb
from commons import fastround_int
from math import ceil,log2


@nb.njit(fastmath=True,parallel=True,error_model='numpy')
def binary_func_search(eval_func,*eval_opts,lb=0.,ub=1.,max_iter=25):
    """ Assume vector is already bounded by the same constraints.
    lb=[0, #_l where D*#_l <=1) and ub=[ #_u > #_l,  #_u <= D*#_l )"""
    low_lam, high_lam = lb, ub
    mid=0.
    for i in range(max_iter):
        mid = 0.5*(low_lam+high_lam)
        s = eval_func(mid,*eval_opts)
        if s>0: high_lam=mid
        elif s<0: low_lam=mid
        else:
            print('Completed prematurely in ', i, 'bisections')
            break
    return mid

@nb.njit(fastmath=True,parallel=True,error_model='numpy')
def power_fixedpop_evalmatch(coef,pop_size,generations,eval_target,power=3.):

    finsum=nb.int64(0)
    for i in nb.prange(1,generations+1):
        finsum+=ceil(coef*(i**power))*pop_size #min evals is pop_size, then increases by it's multiple from there.
    return finsum-eval_target

@nb.njit(fastmath=True,parallel=True,error_model='numpy')
def bestfit_powerresample_coeff(pop_size,generations,eval_target,power=3.):
    """When coef returned: ceil(coef*(current_generation**power)) is the # of times you want to resample your stochastic fitness function."""
    #There sb a
    itr=ceil(log2(generations)*4)
    return binary_func_search(power_fixedpop_evalmatch,pop_size,generations,eval_target,power,lb=0.,ub=1./generations,max_iter=itr)



