import numba as nb
import gopt.commons as cmn
from math import ceil,log2


@nb.njit(**cmn.nb_pcs())
def binary_func_search(lb,ub,max_iter,eval_func,*eval_opts):
    low_lam, high_lam = lb, ub
    mid=0.
    ls=0
    rc=0
    for _ in range(max_iter):
        mid = 0.5*(low_lam+high_lam)
        s = eval_func(mid,*eval_opts)
        if s>0: high_lam=mid
        elif s<0: low_lam=mid
        else:
            #print('Completed prematurely in ', i, 'bisections')
            break
        if ls==s:
            rc+=1
            if rc>=10: #as it could be fitting continuous or discrete, this is also a valid premature stop.
                break
        else:
            rc=0
        ls=s
    return mid

@nb.njit(**cmn.nb_cs())
def power_fixedpop_evalmatch(coef,pop_size,generations,eval_target,power=3.):
    finsum=nb.int64(0)
    jp=generations**power #((i**power)/jp) to preserve significant digits
    for i in nb.prange(1,generations+1):
        finsum+=ceil(coef*((i**power)/jp))*pop_size #min evals is pop_size, then increases by it's multiple from there.
    return finsum-eval_target


@nb.njit(cache=False,**cmn.nb_pcs())
def bestfit_powerresample_coeff(pop_size,generations,eval_target,power=3.): #assume its at least >1.
    """When coef returned: ceil(coef*(current_generation**power)/(f_gen**power)) is the # of times you want to resample your stochastic fitness function.
    Use this format to preserve significant digits which might be important for the rounding."""
    #There sb a
    itr=ceil(log2(generations)*20)
    return binary_func_search(0.,eval_target-pop_size,itr,power_fixedpop_evalmatch,pop_size,generations,eval_target,power)



