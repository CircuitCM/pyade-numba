import numpy as np
import numba as nb
from scipy.stats._qmc import Sobol
import gopt.config as cfg
from math import ceil,floor,log,log2,log1p
import random as rand
from typing import Callable, Union, List, Tuple, Any
from numba import njit, types
from numba.core.extending import overload, register_jitable as rg, register_jitable


#extra common overload stuff
def _ftpop(op,tp,i):return tp if tp is None else tp[i,:op].reshape((1,op)) if type(i) is int else tp[:,:op]
@overload(_ftpop,**cfg.jit_s)
def _ftpop_(op,tp,i):
    if isinstance(tp,types.Array):
        if isinstance(i,types.Integer):
            def _p(op,tp,i): return tp[i,:op].reshape((1,op))
        else:
            def _p(op, tp,i):return tp[:, :op]
    else:
        def _p(op,tp,i): pass
    return _p

@nb.njit(**cfg.jit_s)
def _ss(f):
    rand.seed(f)
    np.random.seed(f)

def set_seed(seed):
    if seed is not None:
        _ss(seed)
        rand.seed(seed)
        np.random.seed(seed)
@overload(set_seed,**cfg.jit_s,inline='always')
def _rset_(f):
    if isinstance(f,types.NoneType):
        def _f(f): pass
    else:
        def _f(f): return rand.seed(f)
    return _f

#can be used for CR too
def _fset(f):return rand.uniform(f[0], f[1]) if type(f) is np.ndarray else f
@overload(_fset,**cfg.jit_s,inline='always')
def _fset_(f):
    if isinstance(f,types.Array):
        def _f(f): return rand.uniform(f[0], f[1])
    else:
        def _f(f): return f
    return _f

def _pset(p):return rand.randrange(p[0], p[1]) if type(p) is np.ndarray else p
@overload(_pset,**cfg.jit_s)
def _pset_(p):
    if isinstance(p,types.Array):
        def _p(p): return rand.randrange(p[0], p[1]) #jitter p for pbest selection by per generation.
    else:
        def _p(p): return p
    return _p

def _meval(m,*args):
    if m is not None:m(*args)
@overload(_meval,**cfg.jit_s) #allowing parallel tho idk what it could end up doing to it if monitor isn't. Nothing, the monitor func can be parallel.
def _meval_(m,*args):
    if isinstance(m,types.Callable):
        def _m(m,*args): m(*args)
    else:
        def _m(m,*args): pass
    return _m

#Generational ema
def disc_orn(v, c): return None if v is None else v * c
@overload(disc_orn, **cfg.jit_s)
def _disc_orn(v, c):return lambda v, c: None if isinstance(v, types.NoneType) else lambda v, c: v * c

def optgen_ema(v, c, em_r, n_r):
    if v is None:
        return (1. - c) * em_r + c * n_r, None
    else:
        em_r = (1. - v) * em_r + v * n_r
        return em_r, c

@overload(optgen_ema, **cfg.jit_s)
def _optgen_ema(v, c, em_r, n_r):
    if isinstance(v, types.NoneType):
        def cl(v, c, em_r, n_r):
            return (1. - c) * em_r + c * n_r, None
    else:
        def cl(v, c, em_r, n_r):
            em_r = (1. - v) * em_r + v * n_r
            return em_r, c
    return cl


@rg(**cfg.jit_s)
def knownfitness_fixedeval_logcost(evals, fitness, maxevals, minfit, maxfit):
    return log1p(evals)/log1p(maxevals) + log1p(max(fitness,minfit)-minfit)/log1p(maxfit-minfit) -1
    #The assumption is that as optimizer gets closer to beating the fitness, the rate of improvement will decrease roughly proportionally.
    #if it worsens h-param tuning performance try linear version below.

@rg(**cfg.jit_s)
def knownfitness_fixedeval_lincost(evals, fitness, maxevals, minfit, maxfit):
    return (evals/maxevals) + (max(fitness,minfit)-minfit)/(maxfit-minfit) - 1

@rg(**cfg.jit_s)
def logfit_lineval_cost(evals, fitness, maxevals, minfit, maxfit):
    #So linear evals does make more sense, confuses the optimizer less alegedly
    return (evals/maxevals) + log1p(max(fitness,minfit)-minfit)/log1p(maxfit-minfit) -1


@rg(**cfg.jit_s)
def count_monitor(mute_pop:np.ndarray,ct_arr:np.ndarray):
    #An example, args should be altered for different optimizers.
    ct_arr[0]+=mute_pop.shape[0]
    ct_arr[1]+=1


@nb.njit(**cfg.jit_s)
def keep_bounds(population: np.ndarray,
                bounds: np.ndarray) -> np.ndarray:
    """
    Constrains the population to its proper limits.
    Any value outside its bounded ranged is clipped.
    :param population: Current population that may not be constrained.
    :type population: np.ndarray
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.ndarray
    :rtype np.ndarray
    :return: Population constrained within its bounds.
    """
    ps, pd = population.shape[0], population.shape[1]
    for v in range(ps * pd):
        i = v // pd
        n = v % pd
        #Past experience has demoed this to be faster than if elif.
        population[i, n] = min(max(population[i, n], bounds[n, 0]), bounds[n, 1])
    return population

@nb.njit(**cfg.jit_tp)
def keep_bounds_pl(population: np.ndarray,
                bounds: np.ndarray) -> np.ndarray:
    ps, pd = population.shape[0], population.shape[1]
    for v in nb.prange(ps * pd):
        i = v // pd
        n = v % pd
        #Past experience has demoed this to be faster than if elif.
        population[i, n] = min(max(population[i, n], bounds[n, 0]), bounds[n, 1])
    return population

@nb.njit(**cfg.jit_s)
def place_bounded_vec(search_vec: np.ndarray,
                      param_vec:np.ndarray,
                bounds: np.ndarray) -> np.ndarray:
    for n in range(search_vec.shape[0]):
        param_vec[n] = min(max(search_vec[n], bounds[n, 0]), bounds[n, 1])
    return param_vec


@nb.njit(**cfg.jit_s)
def bounds_safe(p_vec: np.ndarray,bounds: np.ndarray):
    sb=np.int64(0)
    for v in range(p_vec.shape[0]):
        if not (bounds[v,0]<=p_vec[v]<=bounds[v,1]):
            sb+=1
    #counts how many parameters are not within bounds, as there is no other preference for infeasibles besides falling within constraints.
    #Could also turn it into lnorm_bounds_cost but the intention is to maximize parameter search within the feasible set
    #Which in turn minimizes the # of search space scalars than need to be truncated to the boundary edges in the parameter space.
    #bounds_safe count is the invariant metric comparison that should satisfy this when minimized.
    #While this is used for rejection sampling, the crossover rng is calculated only once to remove bias against searching some dimensions
    #more than others.
    return sb

#Check which is quicker
def _scaled_bc(n,pv,scale)->float:pass
@overload(_scaled_bc, **cfg.jit_s)
def _scaled_bc_(n,pv,scale):
    if isinstance(scale, types.NoneType):
        def _impl(n,pv,scale): return pv
    elif isinstance(scale, types.Number):
        def _impl(n,pv, scale):return pv*scale
    elif isinstance(scale, types.Array):
        def _impl(n,pv, scale):return pv*scale[n]
    return _impl

@register_jitable(**cfg.jit_s)
def ri64(rd:nb.float64):
    return nb.int64(rd+.5)

@nb.njit(**cfg.jit_s)
def lnorm_bounds_cost(population: np.ndarray,
                      bounds: np.ndarray,
                      bc: np.ndarray, #total cost for population member.
                      scale:None|np.ndarray|float|Any=True,
                      norm:float=2.): #None is
    #Scale, None then unit distance, float then constant multiple, array dim dependent multiple.
    #Recommended you provide a scaling array for each dimension, see examples. diff bounds will use redundant cpu cycles.
    ps, pd = population.shape[0], population.shape[1]
    bc[:]=0
    for v in range(ps * pd):
        i = v // pd
        n = v % pd
        pv=abs(population[i,n] - min(max(population[i,n], bounds[n,0]),  bounds[n,1]))
        #pv=max(0, bounds[n,0] - population[i,n]) + max(0, population[i,n] - bounds[n,1]) #see which is faster.
        bc[i] += _scaled_bc(n,pv,scale)**norm
    return bc


@nb.njit(**cfg.jit_s)
def place_bounded_vec_calccost(search_vec: np.ndarray,
                               param_vec:np.ndarray,
                               bounds: np.ndarray,
                               #bc: np.ndarray,  # total cost for population member.
                               scale: None | np.ndarray | float | Any = None,
                               norm: float = 2.
                               ):
    """
    Enforces bounds to parameter vector and calculates total bounds violation cost of vector like lnorm_bounds_cost, but for a vec.
    :param search_vec:
    :param param_vec:
    :param bounds:
    :param bc:
    :param scale:
    :param norm:
    :return:
    """
    bc=0.
    for n in range(search_vec.shape[0]):
        sp=search_vec[n]
        pp = min(max(sp, bounds[n, 0]), bounds[n, 1])
        bc+=_scaled_bc(n,abs(pp-sp),scale)**norm
        param_vec[n] = pp
    return bc



def make_bounds_costscalingarray(bounds:np.ndarray,p:float|np.ndarray=2.):
    if isinstance(p,float):
        scr=np.empty((bounds.shape[0],),dtype=np.float64)
        scr[:]=p
    else: scr=p
    scr/=(bounds[:,1]-bounds[:,0])
    return scr


def mk_fitness_thresh_stop(fitn:float=0.):
    if fitn is not None:
        #hopefully..., yer seems to work.
        @nb.njit(**cfg.jit_s)
        def f(population,fitness):
            return fitness_threshold_stop(population,fitness,fitn)
        return f
    return None

@nb.njit(**cfg.jit_s)
def fitness_threshold_stop(population:np.ndarray,fitness:np.ndarray,stopt:float=0.):
    for i in fitness:
        if i<stopt:
            return True
    return False

#Add coordinate based pop stopper.

@rg(**cfg.jit_tp)
def initpop_randuniform(population_size, individual_size, bounds,*_rec)-> np.ndarray:
    minimum = bounds[:,0]
    maximum = bounds[:,1]
    population = np.random.rand(population_size, individual_size) * (maximum-minimum) + minimum
    return population


@nb.njit(**cfg.jit_tp)
def placepop_randuniform(pop, bounds,*_rec):
    ps, pd = pop.shape[0], pop.shape[1]
    if ps*pd>2000:
        for v in nb.prange(ps*pd):
            i = v // pd
            n = v % pd
            pop[i, n] = rand.uniform(bounds[n, 0], bounds[n, 1])
    else:
        for v in range(ps*pd):
            i = v // pd
            n = v % pd
            pop[i,n]=rand.uniform(bounds[n,0],bounds[n,1])


@register_jitable(**cfg.jit_s)
def place_randnormal(vec,mean,std):
    for i in range(vec.shape[0]):
        vec[i] = rand.normalvariate(mean, std)


@register_jitable(**cfg.jit_s)
def place_randnormal_fbds(vec,mean,std,minn,maxx):
    for i in range(vec.shape[0]):
        vec[i] = min(max(rand.normalvariate(mean, std),minn),maxx)


def init_soboluniform(population_size, individual_size, bounds)-> np.ndarray:
    sob_engine=Sobol(individual_size)
    minimum = bounds[:,0]
    maximum = bounds[:,1]
    population = sob_engine.random(population_size) * (maximum-minimum) + minimum
    return population


def init_population(population_size: int, individual_size:int,
                    bounds: np.ndarray,rand_spec: Callable|str|np.ndarray='sobol') -> np.ndarray:
    """
    Creates a random population within its constrained bounds.
    :param population_size: Number of individuals desired in the population.
    :type population_size: int
    :param individual_size: Number of features/gens.
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: Union[np.ndarray, list]
    :rtype: np.ndarray
    :return: Initialized population.
    """
    individual_size=individual_size if type(bounds) is not np.ndarray else bounds.shape[0] if type(individual_size) is not int else individual_size #dont need the second if actually eh
    if type(rand_spec) is str:
        if rand_spec=='sobol':
            _p=init_soboluniform(population_size, individual_size, bounds) #most effective if pop_size = 2**k for some k>2. Otherwise won't be that different from uniform.
        elif rand_spec=='uniform':
            _p = initpop_randuniform(population_size, individual_size, bounds)
        else:
            _p = init_soboluniform(population_size, individual_size, bounds)
    elif isinstance(rand_spec,Callable):
        _p=rand_spec(population_size, individual_size, bounds)
    else: #assume it's initialized numpy array.
        _p=rand_spec
    # if bounds is not None:
    #     keep_bounds(_p,bounds)

    return _p


#So current template is:
#ci:np.int64, m_pop: np.ndarray,pop: np.ndarray,idx:np.ndarray, *args
#ci is the current index, because of crossovers it is always utilized.
#
#*args : best idxs, cr, crw, f, fw, h etc go here.

@nb.njit(**cfg.jit_s)
def bin_mutate(crr:np.ndarray, ci:np.int64, m_pop: np.ndarray, pop: np.ndarray, idx:np.ndarray, f):
    #Could create duplicates... though unlikely with crossover
    for i in crr:
        m_pop[i]=f * (pop[idx[0],i] - pop[idx[1],i]) + pop[idx[2],i]



@nb.njit(**cfg.jit_s)
def c_to_best_2_mutate(crr:np.ndarray, ci:np.int64, m_pop_vec: np.ndarray, pop: np.ndarray, idx:np.ndarray, b_idx, f):
    """
    Calculates the mutation of a vector based on the
    "current to best/2/bin" mutation. This is
    V_{i, G} = X_{i, G} + F * (X_{best, G} - X_{i, G} + F * (X_{r1, G} - X_{r2, G} Actually I think it's wrong lol.
    Also includes binary crossover.
    """
    for i in crr:
        # uniform first seems to be quicker than checking, so in theory only quicker to switch conditional check order if cr<1/n... which is rare for high dim.
        m_pop_vec[i]=pop[ci,i] + f * (pop[b_idx,i] - pop[ci,i]) + f * (pop[idx[0],i] - pop[idx[1],i])



def _get_br(ci, b_idx,p_idx):pass
@overload(_get_br, **cfg.jit_s)
def _get_br_(ci, b_idx, p_idx):
    if isinstance(p_idx, types.Array):
        def _impl(ci, b_idx, p_idx): return b_idx[rand.randint(0, p_idx[ci])]
    elif isinstance(p_idx,types.Integer):
        def _impl(ci, b_idx, p_idx): return b_idx[rand.randint(0,p_idx)]
    else:
        raise TypeError("Unsupported type for p_idx in _get_br")
    return _impl

def _get_f(ci,i, f):pass #already handles the 1d case.
@overload(_get_f, **cfg.jit_s)
def _get_f_(ci,i, f):
    if isinstance(f, types.Array):
        if f.ndim==2:
            def _impl(ci,i, f):return f[ci,i]
        else:
            def _impl(ci,i, f): return f[ci]
    elif isinstance(f, types.Float):
        def _impl(ci,i, f): return f
    else:
        raise TypeError("Unsupported type for f in _get_f")
    return _impl


@nb.njit(**cfg.jit_s)
def c_to_pbest_mutate(crr:np.ndarray, ci:np.int64, m_pop_vec: np.ndarray, pop: np.ndarray, idx:np.ndarray, b_idx:np.ndarray, p:np.ndarray, f:np.ndarray):
    """
    Calculates the mutation of a vector based on the
    "current to p-best" mutation. Includes binary crossover. This is
    V_{i, G} = X_{i, G} + F * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}.
    For pbest, its ok if the bp selected is the current vector or one of the randoms, only that the randoms aren't equal.
    What is nice about pbest is that if ci = br then it becomes ~rand1.
    """
    #Later on consider mutations that select dimensions randomly, this sb a simple matter by using randint for an idx range.
    br = _get_br(ci, b_idx, p)
    for i in crr:
        scale = _get_f(ci,i, f)
        m_pop_vec[i]=pop[ci,i] + scale * (pop[br,i] - pop[ci,i]) + scale * (pop[idx[0],i] - pop[idx[1],i])


@nb.njit(**cfg.jit_s)
def c_to_rand1_mutate(crr:np.ndarray, ci:np.int64, m_pop_vec: np.ndarray, pop: np.ndarray, idx:np.ndarray, f:np.ndarray, k:np.ndarray):
    """
    Calculates the mutation of a vector based on the
    "current to rand/1" mutation. crr are the crossover indexes, m_pop_vec is already inited with pop.
    U_{i, G} = X_{i, G} + K * (X_{r1, G} - X_{i, G} + F * (X_{r2. G} - X_{r3, G}
    """
    for i in crr:
        s1 = _get_f(ci,i, f)
        s2 = _get_f(ci,i, k)
        m_pop_vec[i]=pop[ci,i]+ s2 * (pop[idx[0],i] - pop[ci,i]) + s1 * (pop[idx[1],i] - pop[idx[2],i])



@nb.njit(**cfg.jit_s)
def c_to_pbestw_mutate(crr:np.ndarray, ci:np.int64, m_pop_vec: np.ndarray, pop: np.ndarray, idx:np.ndarray, b_idx:np.ndarray, p: np.int64 | np.ndarray, f:np.ndarray, f_w:np.ndarray):
    """
    Calculates the mutation of a vector based on the
    "current to p-best weighted" mutation. This is
    V_{i, G} = X_{i, G} + F_w * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    Also includes binary crossover.
    """
    #p is not a vector in this one but idk
    br= _get_br(ci, b_idx, p)
    for i in crr:
        m_pop_vec[i] = pop[ci,i]
        s1 = _get_f(ci,i, f)
        s2 = _get_f(ci,i, f_w)
        m_pop_vec[i]+=s2 * (pop[br,i] - pop[ci,i]) + s1 * (pop[idx[0],i] - pop[idx[1],i])


@nb.njit(**cfg.jit_s)
def c_to_pbesta_mutate(crr:np.ndarray, ci:np.int64, m_pop_vec: np.ndarray, pop: np.ndarray, idx:np.ndarray, b_idx:np.ndarray, p:np.ndarray, f:np.ndarray,a:np.ndarray):
    """
    Calculates the mutation of a vector based on the
    "current to p-best-archive" mutation. This is
    V_{i, G} = X_{i, G} + F * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    Where r2 comes from the union of the current population and the archive.
    This essentially turns it into a double layered pbest, the second pbest layer is the population itself.
    For efficient implementation, we know that r2 or idx[1] has pop_size - 2 = ps possibilities.
    We also know that the archive has a.shape[0] = ars possibilities, to simulate a uniform choice
    from P U A select : pop[idx[1]] if ps/(ps+as) < random.uniform(0,1) else a[rand.randrange(0,ars)]
    """
    #Later on consider mutations that select dimensions randomly, this sb a simple matter by using randint for an idx range.
    #Turned out to be a shockingly easy add, nothing else needed to generalize the current setup.
    br = _get_br(ci, b_idx, p)
    ps,ars=pop.shape[0]-2,a.shape[0]
    rand2 = pop[idx[1]] if ps/(ps+ars) < rand.random() else a[rand.randrange(0,ars)]
    for i in crr:
        scale = _get_f(ci,i, f)
        m_pop_vec[i]=pop[ci,i] + scale * (pop[br,i] - pop[ci,i]) + scale * (pop[idx[0],i] - rand2[i])


_N=types.none


#going to remove this, up to user to enforce parameters instead
def enforce_bounds(population: np.ndarray,
                   bounds: np.ndarray, enf_bds):pass
@overload(enforce_bounds, **cfg.jit_s)
def _enforce_bounds(population, bounds,enf_bds):
    if enf_bds is _N:
        def _eb(population, bounds,enf_bds):
            pass
    else:
        def _eb(population, bounds,enf_bds):
            keep_bounds(population,bounds)
    return _eb


@nb.njit(**cfg.jit_s)
def bin_cross_init(m_pop_vec:np.ndarray,pop_vec:np.ndarray,idxg:np.ndarray,ci,cr): #I'll assume cr is still an int for now. add array overload if it is happened on.
    sh=idxg.shape[0]
    l=0
    j_rand = rand.randint(0, idxg.shape[0] - 1)
    for i in range(0,sh):
        if rand.uniform(0., 1.) < _get_f(ci,i,cr) or i == j_rand:
            idxg[l]=i
            l+=1
        else:
            m_pop_vec[i]=pop_vec[i]
    return idxg[:l]


#uchoice_mutator : unique choice mutator, incorporates the current-member-skipping random choice
@nb.njit(**cfg.jit_s) #Comment this out and it should still be runnable in python-mode up to individual crossover and mutation. But this is only for bug testing, should never need python mode for the uchoice_mutator
def uchoice_mutator(population: np.ndarray,
                    m_pop:np.ndarray,
                    cr: float|np.ndarray,
                    bounds: np.ndarray,
                    reject_mx:int, #None or int, None will use the smaller no-resampling compilation. ~5 is a good balance when optimal fitness is along boundaries.
                    cross_apply,  #these should probably have underscores... but lets user implement their own.
                    mut_apply,
                    _ns: np.int64,  #number of r selections for mutation operator
                    _idx: np.ndarray,  #should already be initialized (thread count, len_pop) with ints [0,n) for thread parallelism.
                    _crossgen:np.ndarray,
                    _t_pop:np.ndarray,  #temporary population holder, needed to replace the best infeasible.
                    *mut_args): #best idxs, p, k, f, fw, h etc go here for dynamic compile.
    #Rejection sampling is a "closest to feasible" sampler to diminish bias, but the cost is ~ O(C*pop_size*pop_dims*num_resamples), so can become expensive.
    #So the user still needs to enforce actually infeasible bounds in the parameter space, so that optimizer is unconstrained in the search space. (unless they enforce_bounds in search space, not recommended)
    #Then add a fitness discount to eval function. Or enable them in the search space in config. Though some optimizers like lshade-cnepsin don't respect them.

    #if pop_size*pop_dims*num_resamples< 100k to 500k then probably not worth parallel, the overhead of launching parallel threads can be like 5-10x a couple hundred ops..

    _uchoice_loop(population, m_pop, bounds, reject_mx, cr, cross_apply, mut_apply, _ns, _idx, _crossgen, _t_pop, *mut_args, cfg.nc_t)

    #removed, truncating search scalars never helps DE algos perform better, instead add bounds violation cost to fitness func and
    #use a parameter population array that truncates the search population array with bounds and eval the fitness func with that
    #when the eval parameters are literally infeasible. Make examples.
    #enforce_bounds(m_pop,bounds,enf_bounds)


def _uchoice_loop(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args):
    if reject_mx is None or bounds is None or _t_pop is None: _uchoice_nsample_loop(population, m_pop, cr, cross_apply, mut_apply, _ns, _idx, _crossgen, *mut_args)
    else:_uchoice_rejectbounds_loop(population, m_pop, bounds, reject_mx, cr, cross_apply, mut_apply, _ns, _idx, _crossgen, _t_pop,*mut_args)

#this one does need to be inlined so it sees the new do nothing dispatcher to make the toggle work. Actually maybe not, knowing how it works with non njit funcs
@overload(_uchoice_loop, **cfg.jit_s)#inline='always'
def _uchoice_loop_(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args):
    if reject_mx is _N or bounds is _N or _t_pop is _N:
        def _r(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args):
            _uchoice_nsample_loop(population,m_pop,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,*mut_args)
    #elif _adegen is _N:
    else:
        def _r(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args):
            _uchoice_rejectbounds_loop(population, m_pop, bounds, reject_mx, cr, cross_apply, mut_apply, _ns, _idx, _crossgen, _t_pop, *mut_args)
    # else:
    #     def _r(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,_adegen,*mut_args):
    #         _uchoice_rejectpolicy_loop(population, m_pop, bounds, reject_mx, cr, cross_apply, mut_apply, _ns, _idx, _crossgen, _t_pop,_adegen, *mut_args)
    return _r


@rg(**cfg.jit_tp)
def _uchoice_rejectbounds_loop(population, m_pop, bounds, reject_mx, cr, cross_apply, mut_apply, _ns, _idx, _crossgen, _t_pop, *mut_args):
    pop_size = population.shape[0]
    p_d=population.shape[1]
    reps=reject_mx
    if cfg.parallel_enabled and pop_size*(p_d + 20)*reject_mx//8>1100:
        #If you are being really performance oriented estimate the avg # of rejections from the previous iterations with an extra array for the divisor.
        #Then estimate the likely # of rejections based on pop num, current average cr, and current f.
        nthds=nb.get_num_threads()
        ld=nb.set_parallel_chunksize(ceil(population.shape[0]/nthds)) #consider making a chunk size merging system when total # ops is clearly small enough.
        for v in nb.prange(0, pop_size):
            tid=nb.get_thread_id()
            _reject_bounds_iter(tid,v,_ns,p_d,cross_apply,_t_pop,population,_crossgen,cr,reps,pop_size,_idx,mut_apply,bounds,m_pop,*mut_args[:-1])
        nb.set_parallel_chunksize(ld)
    else:
        tid=0 #if pcfg.parallel_enabled else nb.get_thread_id() #if it lands here but still parallel enabled, it's because cost of launching parallel threads is >, otherwise its because simultaneous optimizations using same array.
        #print('tid',tid)
        for v in range(0, pop_size):
            _reject_bounds_iter(tid,v,_ns,p_d,cross_apply,_t_pop,population,_crossgen,cr,reps,pop_size,_idx,mut_apply,bounds,m_pop,*mut_args[:-1])


@rg(**cfg.jit_s)
def _reject_bounds_iter(tid,v,_ns,p_d,cross_apply,_t_pop,population,_crossgen,cr,reps,pop_size,_idx,mut_apply,bounds,m_pop,*mut_args):
    # hoping these don't cost anything, so it will pick up on it's last memory location.
    _s_mem = np.empty((_ns, 2), dtype=np.int64)  # Yeah does seem a little quicker
    _bb = p_d + 1
    #print(cr)
    _ccr = cross_apply(_t_pop[tid], population[v], _crossgen[tid],v, cr)
    for _ in range(reps):  # This might get more likely to hit with more dimensions so consider a scaling thing, or not.
        _sw_in(_ns, pop_size, v, _idx, tid, _s_mem)
        mut_apply(_ccr, v, _t_pop[tid], population, _idx[tid, :_ns], *mut_args)
        _sw_out(_ns, _idx, tid, _s_mem)
        sb = bounds_safe(_t_pop[tid], bounds)
        if sb < _bb:
            m_pop[v] = _t_pop[tid]  # This and bounds safe are probably what is contributing to the linear delay scaling with # dims.
            _bb = sb
            if sb == 0:
                break


@rg(**cfg.jit_tp)
def _uchoice_nsample_loop(population,m_pop,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,*mut_args):
    pop_size = population.shape[0]
    p_d=population.shape[1]
    if cfg.parallel_enabled and pop_size*(p_d + 20)>1100: #could end up being different for different computers. For high pop count it's still more efficient. hense the +8.
        nthds=nb.get_num_threads()
        ld=nb.set_parallel_chunksize(ceil(population.shape[0]/nthds)) #consider making a chunk size merging system when total # ops is clearly small enough.
        for v in nb.prange(0, pop_size):
            tid=nb.get_thread_id()
            _nsample_iter(tid, v, _ns, p_d, cross_apply, population, _crossgen, cr, pop_size, _idx, mut_apply, m_pop, *mut_args[:-1])
        nb.set_parallel_chunksize(ld)
    else:
        tid = 0 #if pcfg.parallel_enabled else nb.get_thread_id()
        for v in range(0, pop_size):
            _nsample_iter(tid,v,_ns,p_d,cross_apply,population,_crossgen,cr,pop_size,_idx,mut_apply,m_pop,*mut_args[:-1])


@rg(**cfg.jit_s)
def _nsample_iter(tid,v,_ns,p_d,cross_apply,population,_crossgen,cr,pop_size,_idx,mut_apply,m_pop,*mut_args):
    _s_mem = np.empty((_ns, 2), dtype=np.int64)  # Yeah does seem a little quicker
    _bb = p_d + 1
    _ccr = cross_apply(m_pop[v], population[v], _crossgen[tid],v, cr)
    _sw_in(_ns, pop_size, v, _idx, tid, _s_mem)
    mut_apply(_ccr, v, m_pop[v], population, _idx[tid, :_ns], *mut_args)
    _sw_out(_ns, _idx, tid, _s_mem)


#_sw_in and _sw_out are durstenfeld random shuffle with current skipping for j.
@njit(**cfg.jit_s)
def _sw_in(n_shuffles, pop_size, v, _idx, tid, _s_mem):
    for i in range(n_shuffles):
        # Generate a random index j with i <= j < n
        j = rand.randint(i, pop_size - 2)  # we only skip j, i gives us our selection
        j = j + 1 if j >= v else j  # that skip current vector thingy.
        # Record swapped indices
        _s_mem[i, 0] = i
        _s_mem[i, 1] = j
        # Swap in-place
        tmp = _idx[tid, i]
        _idx[tid, i] = _idx[tid, j]
        _idx[tid, j] = tmp

@njit(**cfg.jit_s)
def _sw_out(n_shuffles, _idx, tid, _s_mem):
    for i in range(n_shuffles - 1, -1, -1):
        x, y = _s_mem[i, 0], _s_mem[i, 1]
        tmp = _idx[tid, x]
        _idx[tid, x] = _idx[tid, y]
        _idx[tid, y] = tmp


@nb.njit(**cfg.jit_s)
def select_better(fitness: np.ndarray, new_fitness: np.ndarray,_idxs: np.ndarray) -> np.ndarray:
    """
    Selects the best individuals based on their fitness.
    """
    bt=0
    for i in range(fitness.shape[0]):
        if new_fitness[i]<fitness[i]:
            _idxs[bt]=i
            bt+=1
    return _idxs[:bt]

@nb.njit(**cfg.jit_s)
def bin_rchoose(binr: np.ndarray, prob:float) -> np.ndarray:
    """
    Randomly divide indices between likelihood and not likelihood.
    """
    #Because it's binary it allows for this single pass unique for of assignment. The problem is it's biased by index
    #location, so you should randomize/shuffle any coordinate references to it if it's cheaper
    bt=0
    btn=binr.shape[0]
    for i in range(binr.shape[0]):
        if prob<rand.random():
            binr[bt]=i
            bt+=1
        else:
            btn-=1
            binr[btn]=i
    return binr[:bt],binr[bt:]

@rg(**cfg.jit_s)
def fast_binom(iters,p):
    s=0
    for _ in range(1,iters):
        s+=p>rand.random()
    return s

@nb.njit(**cfg.jit_s)
def durstenfeld_p_shuffle(a, k):
    """
    Perform up to k swaps of the Durstenfeld shuffle on array 'a',
    storing each swap in 'swap_memory' and then reversing them
    to restore 'a' to its original order.
    Given
    """
    n = a.shape[0]
    num_swaps = min(k, n - 1)
    for i in range(num_swaps):
        j = rand.randrange(i,n)
        # Swap in-place
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp


#---- Depricated or experiment with later.


#@nb.njit(**pcfg.jit_tp)
@rg(**cfg.jit_tp)
def _uchoice_rejectpolicy_loop(population, m_pop, bounds, reject_mx, cr, cross_apply, mut_apply, _ns, _idx, _crossgen, _t_pop,_adegen, *mut_args):
    pop_size = population.shape[0]
    p_d=population.shape[1]
    reps=reject_mx #min(reject_mx, max(pcfg.reject_sample_mn, ceil(pcfg.reject_sample_mult * (p_d ** pcfg.reject_sample_dimorder))))
    # if pop_size*pop_dims*num_resamples< 100k to 500k then probably not worth parallel, the overhead of launching parallel threads can be like 5-10x a couple hundred ops..
    nthds = nb.get_num_threads()
    ld = nb.set_parallel_chunksize(ceil(population.shape[0] / nthds))
    for v in nb.prange(0, pop_size):
        tid=nb.get_thread_id()
        _reject_policy_iter(tid,v,_ns,p_d,cross_apply,_t_pop,population,_crossgen,cr,reps,pop_size,_idx,mut_apply,bounds,m_pop,*mut_args[:-1])
    nb.set_parallel_chunksize(ld)

#@nb.njit(**pcfg.jit_s)
@rg(**cfg.jit_s)
def _reject_policy_iter(tid,v,_ns,p_d,cross_apply,_t_pop,population,_crossgen,cr,reps,pop_size,_idx,mut_apply,bounds,m_pop,*mut_args):
    # hoping these don't cost anything, so it will pick up on it's last memory location.
    _s_mem = np.empty((_ns, 2), dtype=np.int64)  # Yeah does seem a little quicker
    _bb = p_d + 1
    _ccr = cross_apply(_t_pop[tid], population[v], _crossgen[tid], cr)
    #for now
    scale=1/(bounds[:,1]-bounds[:,0])
    for _ in range(reps):  # This might get more likely to hit with more dimensions so consider a scaling thing, or not.
        _sw_in(_ns, pop_size, v, _idx, tid, _s_mem)
        mut_apply(_ccr, v, _t_pop[tid], population, _idx[tid, :_ns], *mut_args)
        _sw_out(_ns, _idx, tid, _s_mem)
        sb = _rejection_policy(_t_pop[tid],population,v, bounds,scale)
        if sb < _bb:
            m_pop[v] = _t_pop[tid]  # This and bounds safe are probably what is contributing to the linear delay scaling with # dims.
            _bb = sb#now best of reps random samples

#Old experiment to see if it's possible to improve global convergence in high dimensions with low population count. By biasing
#population coordinates to relative diagonal locations and unique positions. Idea was to eliminate degenerate dimensions which
#occur when a single parameter between all the individuals are nearly identical, so that std~=0. This makes it impossible for new
#exploration to occur along the axis, eliminating a potentially significant range of possible solutions.
#However this policy doesn't seem to improve high dimensional convergence on a harder than normal rastrigin, it will still end up getting "stuck"
#Though a high difficulty rastrigin is still an entirely seperable function, it's possible that on non-separable functions this policy
#could actually help, but I don't have the time to experiment with it right now.

#Potentially more effective methods to address degenerate dimensions: get pop mean and std, generate hyper sphere from it, then
#select samples that minimize the sqrd error of the population regressed onto the h-sphere. O(m) would be just the current onto existing sphere,
#as an individual, sqrd error or abs error or another error metric would matter less, if O(n*m) the complete h-sphere and pop error recalc
#Here the error order would matter significantly, as well as possible hyper-parameters to tune how relaxed it should be... idk
#The next idea is every n-iterations (maybe every one), check if a scaled parameter's population breadth is an order of magnitude
#(using a h-parameter between 0 and 1) smaller than the mean/median or max breadth scaled parameter, if yes then pick a random or
#intentional population member(s) and perturb that parameter with a normal or uniform with scaling ~ comparison breadth metric.
#or instead, keep an archive of old members, going down the archive in historical order, select the first that provides the
#required parameter breadth, or if none do then the closest one in the archive. As this might not promote enough diversity, you can
#instead select the first n or closest n, then randomly choose from that sub group.
@njit(**cfg.jit_s)
def _rejection_policy(m_vec: np.ndarray,
                     pop:np.ndarray,
                     v:int,#The index of the current vector so it can be skipped
                     bounds: np.ndarray,
                     scale: None | np.ndarray | float,
                     #_dim_degenscale:float, #calculated once at onset of mutator
                     #_prox_rank_scale:float,
                     _bounds_norm: float = 2.):
    """
    Test policy with multiple metrics to see evolution of rejection sampling.
    """
    pd=pop.shape[0]
    ivd=pop.shape[1]
    bc=0.
    mean_p=np.zeros((ivd,),dtype=np.float64)
    #mean_p should already be adjusted without current vector using n=pop.shape[0], (n*x_mean - x_curr)/(n-1).
    #While I'm deciding I'll just make mean_p here then change for the actual implementation if it's the best metric.
    for i in range(pd):
        if v == i:
            for n in range(ivd):
                mean_p[n]+=m_vec[n]
        else:
            for n in range(ivd):
                mean_p[n]+=pop[i,n]
    for n in range(ivd):
        mean_p[n]/=pd
        sp=m_vec[n]
        pp = min(max(sp, bounds[n, 0]), bounds[n, 1])
        bc+=(abs(pp-sp)*scale[n])**_bounds_norm
        #mean_p[n]=((pd-1)*mean_p[n]+sp)/pd #for later if best metric.
    mxr = 0. #max proximity rank
    tr=0. #total rank
    l_meff=0. #otherwise it could create a discount from asymmetric distances between current and other vectors, we don't want that, only want average/total dim dispersion, but compare
    l2t=0.
    l1t=0.
    ml2t=0.
    ml1t=0.

    id=ivd**.5
    bc/=id
    od=((pd-1)*ivd)**.5
    mod=(pd*ivd)**.5
    pxr=1/(pd-1)
    for i in range(pd):
        if v == i:
            for n in range(ivd):
                ml1=abs(m_vec[n]-mean_p[n])*scale[n]
                ml1t+=ml1
                ml2t+=ml1*ml1
        else:
            #see if it gets unrolled, otherwise
            # i = v // pd
            # n = v % pd
            l10=0.
            l20=0.
            for n in range(ivd):
                l1=abs(pop[i,n]-m_vec[n])*scale[n]
                ml1=abs(pop[i,n]-mean_p[n])*scale[n]
                l10+=l1
                l20+=l1*l1
                ml1t+=ml1
                ml2t+=ml1*ml1
            l1t+=l10
            l2t+=l20
            l20=l20**.5
            l_meff+=(l10-l20*id)/(l20-l20*id)
            l20=1./max(l20,1e-8)
            tr+=l20
            if mxr<l20:
                mxr=l20
    l_meff/=(pd-1)
    prox_rank=((mxr/max(tr,1e-8)) - pxr)/(1-pxr) #lower is better, the least it could be is 1/(pop.shape[0]-1)
    l2t=l2t**.5
    dim_eff=(l1t-l2t*od)/(l2t-l2t*od)
    ml2t=ml2t**.5
    #print(ml2t,ml1t)
    m_dim_eff=(ml1t-ml2t*mod)/(ml2t-ml2t*mod)
    #print('Bounds cost:',bc,'Mean Vector Dim Eff',l_meff,'Total Scalar Dim Eff',dim_eff,'Population Mean Dim Eff',m_dim_eff,'Proximity Eff',prox_rank)
    a=1
    b=log2(pd)#a
    c=.1#1.5
    s=a+b+c
    a/=s
    b/=s
    c/=s
    return bc*a +m_dim_eff*b + prox_rank*c