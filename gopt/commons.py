import numpy as np
import numba as nb
from scipy.stats._qmc import Sobol
import gopt.config as pcfg
from math import ceil,floor
import random as rand
from typing import Callable, Union, List, Tuple, Any
from numba import njit, types
from numba.core.extending import overload


numba_comp_settings=dict(fastmath=True,parallel=True,error_model='numpy')

def nb_cs():
    return numba_comp_settings

def nb_pcs():
    d=numba_comp_settings.copy()
    d.pop('parallel')
    return d

def compile_with_fallback(func):
    try:
        # Attempt to compile in nopython mode (njit)
        return nb.njit(**nb_cs())(func)
    except Exception as e:
        print(f"Failed to compile in nopython mode: {e}")
        # Fallback to object mode
        try:
            return nb.jit(**(nb_cs()|dict(nopython=False)))(func)
        except Exception as e:
            print(f"Failed to compile numba object mode: {e}")
            return func


@nb.njit(**nb_pcs())
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


@nb.njit(**nb_pcs())
def bounds_safe(p_vec: np.ndarray,bounds: np.ndarray):
    sb=np.int64(0)
    for v in range(p_vec.shape[0]):
        if not (bounds[v,0]<=p_vec[v]<=bounds[v,1]):
            sb+=1
    #counts how many parameters are not within bounds, as there is no other preference for infeasibles besides falling within constraints.
    #Could also turn it into lnorm_bounds_cost but the intention is to maximize parameter search within the feasible set
    #minimizing the # of search space scalars than need to be truncated to the boundary edges in the parameter space.
    #bounds_safe count is the invariant metric comparison that should satisfy this when minimizing it.
    #While this is used for rejection sampling the crossover rng is calculated only once to remove bias against searching some dimensions
    #more than others.
    return sb


def _scaled_bc(pop_sclr,bound_vec,scale):
    pass

@overload(_scaled_bc,inline='always')
def _scaled_bc_(pop_sclr,bound_vec,scale):
    if isinstance(scale, types.NoneType):
        def _impl(pop_sclr,bound_vec,scale): return abs(pop_sclr - min(max(pop_sclr, bound_vec[0]),  bound_vec[1]))
    elif isinstance(scale, types.Number):
        def _impl(pop_sclr, bound_vec, scale):return abs(pop_sclr - min(max(pop_sclr, bound_vec[0]), bound_vec[1]))*scale
    else:
        def _impl(pop_sclr,bound_vec,scale): return abs(pop_sclr - min(max(pop_sclr, bound_vec[0]),  bound_vec[1]))/(bound_vec[1]-bound_vec[0])
    return _impl


@nb.njit(**nb_pcs())
def lnorm_bounds_cost(population: np.ndarray,
                      bounds: np.ndarray,
                      bc: np.ndarray, #total cost for population member.
                      scale:None|np.ndarray|float|Any=True,
                      norm:float=2.): #None is
#Scale, None then unit distance, float then constant multiple, array dim dependent multiple, anything else defaults to 1/(diff bounds).
#Recommended you provide a scaling array for each dimension, see examples. diff bounds will use redundant cpu cycles.
    ps, pd = population.shape[0], population.shape[1]
    bc[:]=0
    for v in range(ps * pd):
        i = v // pd
        n = v % pd
        bc[i] += _scaled_bc(population[i, n],bounds[n],scale)**norm
    return bc

def mk_fitness_thresh_stop(fitn:float=0.):
    @nb.njit(inline='always')
    def f(population,fitness):
        return fitness_threshold_stop(population,fitness,fitn)
    return f

@nb.njit(**nb_pcs())
def fitness_threshold_stop(population:np.ndarray,fitness:np.ndarray,stopt:float=0.):
    for i in fitness:
        if i<stopt:
            return True
    return False


def init_randuniform(population_size, individual_size, bounds):
    minimum = bounds[:,0]
    maximum = bounds[:,1]
    population = np.random.rand(population_size, individual_size) * (maximum-minimum) + minimum
    return population

def init_soboluniform(population_size, individual_size, bounds):
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
    individual_size=individual_size if type(bounds) is not np.ndarray else bounds.shape[1] if type(individual_size) is not int else individual_size #dont need the second if actually eh
    if rand_spec=='sobol':
        _p=init_soboluniform(population_size, individual_size, bounds) #most effective if pop_size = 2**k for some k>2. Otherwise won't be that different from uniform.
    elif rand_spec=='uniform':
        _p = init_randuniform(population_size, individual_size, bounds)
    elif isinstance(rand_spec,Callable):
        _p=rand_spec(population_size, individual_size, bounds)
    else: #assume it's initialized numpy array.
        _p=rand_spec
    # if bounds is not None:
    #     keep_bounds(_p,bounds)

    return _p


# def apply_fitness(population: np.ndarray,
#                   func: Callable[[np.ndarray], float],
#                   opts: Any) -> np.ndarray:
#     """
#     Applies the given fitness function to each individual of the population.
#     :param population: Population to apply the current fitness function.
#     :type population: np.ndarray
#     :param func: Function that is used to calculate the fitness.
#     :type func: np.ndarray
#     :param opts: Optional parameters for the fitness function.
#     :type opts: Any type.
#     :rtype np.ndarray
#     :return: Numpy array of fitness for each individual.
#     """
#     if opts is None:
#         return np.array([func(individual) for individual in population])
#     else:
#         return np.array([func(individual, opts) for individual in population])


# def __parents_choice(population: np.ndarray, n_parents: int) -> np.ndarray:
#     pob_size = population.shape[0]
#     choices = np.indices((pob_size, pob_size))[1]
#     mask = np.ones(choices.shape, dtype=bool)
#     np.fill_diagonal(mask, 0)
#     choices = choices[mask].reshape(pob_size, pob_size - 1)
#     parents = np.array([np.random.choice(row, n_parents, replace=False) for row in choices])
#
#     return parents


# def binary_mutation(population: np.ndarray,
#                     f: Union[int, float],
#                     bounds: np.ndarray) -> np.ndarray:
#     """
#     Calculate the binary mutation of the population. For each individual (n),
#     3 random parents (x,y,z) are selected. The parents are guaranteed to not
#     be in the same position than the original. New individual are created by
#     n = z + F * (x-y)
#     :param population: Population to apply the mutation
#     :type population: np.ndarray
#     :param f: Parameter of control of the mutation. Must be in [0, 2].
#     :type f: Union[int, float]
#     :param bounds: Numpy array of tuples (min, max).
#                    Each tuple represents a gen of an individual.
#     :type bounds: np.ndarray
#     :rtype: np.ndarray
#     :return: Mutated population
#     """
#     # If there's not enough population we return it without mutating
#     if len(population) <= 3:
#         return population
#
#     # 1. For each number, obtain 3 random integers that are not the number
#     parents = __parents_choice(population, 3)
#     # 2. Apply the formula to each set of parents
#     mutated = f * (population[parents[:, 0]] - population[parents[:, 1]])
#     mutated += population[parents[:, 2]]
#
#     return keep_bounds(mutated, bounds)

#So current template is:
#ci:np.int64, m_pop: np.ndarray,pop: np.ndarray,idx:np.ndarray, *args
#ci is the current index, because of crossovers it is always utilized.
#
#*args : best idxs, cr, crw, f, fw, h etc go here.

@nb.njit(**nb_pcs())
def bin_mutate(crr:np.ndarray, ci:np.int64, m_pop: np.ndarray, pop: np.ndarray, idx:np.ndarray, f):
    for i in crr:
        m_pop[i]=f * (pop[idx[0],i] - pop[idx[1],i]) + pop[idx[2],i]


# def current_to_best_2_binary_mutation(population: np.ndarray,
#                                       population_fitness: np.ndarray,
#                                       f: Union[int, float],
#                                       bounds: np.ndarray) -> np.ndarray:
#     """
#     Calculates the mutation of the entire population based on the
#     "current to best/2/bin" mutation. This is
#     V_{i, G} = X_{i, G} + F * (X_{best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
#     :param population: Population to apply the mutation
#     :type population: np.ndarray
#     :param population_fitness: Fitness of the given population
#     :type population_fitness: np.ndarray
#     :param f: Parameter of control of the mutation. Must be in [0, 2].
#     :type f: Union[int, float]
#     :param bounds: Numpy array of tuples (min, max).
#                    Each tuple represents a gen of an individual.
#     :type bounds: np.ndarray
#     :rtype: np.ndarray
#     :return: Mutated population
#     """
#     # If there's not enough population we return it without mutating
#     if len(population) < 3:
#         return population
#
#     # 1. We find the best parent
#     best_index = np.argmin(population_fitness)
#
#     # 2. We choose two random parents
#     parents = __parents_choice(population, 2)
#     mutated = population + f * (population[best_index] - population)
#     mutated += f * (population[parents[:, 0]] - population[parents[:, 1]])
#
#     return keep_bounds(mutated, bounds)


@nb.njit(**nb_pcs())
def c_to_best_2_bin_mutate(crr:np.ndarray, ci:np.int64, m_pop_vec: np.ndarray, pop: np.ndarray, idx:np.ndarray, b_idx, f):
    """
    Calculates the mutation of a vector based on the
    "current to best/2/bin" mutation. This is
    V_{i, G} = X_{i, G} + F * (X_{best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    Also includes binary crossover.
    """
    for i in crr:
        # uniform first seems to be quicker than checking, so in theory only quicker to switch conditional check order if cr<1/n... which is rare for high dim.
        m_pop_vec[i]=pop[ci,i] + f * (pop[b_idx,i] - pop[ci,i]) + f * (pop[idx[0],i] - pop[idx[1],i])



# def current_to_pbest_mutation(population: np.ndarray,
#                               population_fitness: np.ndarray,
#                               f: List[float],
#                               p: Union[float, np.ndarray, int],
#                               bounds: np.ndarray) -> np.ndarray:
#     """
#     Calculates the mutation of the entire population based on the
#     "current to p-best" mutation. This is
#     V_{i, G} = X_{i, G} + F * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
#     :param population: Population to apply the mutation
#     :type population: np.ndarray
#     :param population_fitness: Fitness of the given population
#     :type population_fitness: np.ndarray
#     :param f: Parameter of control of the mutation. Must be in [0, 2].
#     :type f: Union[int, float]
#     :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
#     :type p: Union[int, float, np.ndarray]
#     :param bounds: Numpy array of tuples (min, max).
#                    Each tuple represents a gen of an individual.
#     :type bounds: np.ndarray
#     :rtype: np.ndarray
#     :return: Mutated population
#     """
#     # If there's not enough population we return it without mutating
#     if len(population) < 4:
#         return population
#
#     # 1. We find the best parent
#     p_best = []
#     for p_i in p:
#         best_index = np.argsort(population_fitness)[:max(2, int(round(p_i*len(population))))]
#         p_best.append(np.random.choice(best_index))
#
#     p_best = np.array(p_best)
#     # 2. We choose two random parents
#     parents = __parents_choice(population, 2)
#     mutated = population + f * (population[p_best] - population)
#     mutated += f * (population[parents[:, 0]] - population[parents[:, 1]])
#
#     return keep_bounds(mutated, bounds)


def _get_br(ci, f):pass
@overload(_get_br)
def _get_br_(ci, b_idx, p_idx):
    if isinstance(p_idx, types.Array):
        def _impl(ci, b_idx, p_idx): return b_idx[rand.randint(0, p_idx[ci])]
    elif isinstance(p_idx,types.Integer):
        def _impl(ci, b_idx, p_idx): return b_idx[rand.randint(0,p_idx)]
    else:
        raise TypeError("Unsupported type for p_idx in _get_br")
    return _impl

def _get_f(ci, f):pass
@overload(_get_f)
def _get_f_(ci, f):
    if isinstance(f, types.Array):
        def _impl(ci, f): return f[ci]
    elif isinstance(f, types.Float):
        def _impl(ci, f): return f
    else:
        raise TypeError("Unsupported type for f in _get_f")
    return _impl


@nb.njit(**nb_pcs())
def c_to_pbest_mutate(crr:np.ndarray, ci:np.int64, m_pop_vec: np.ndarray, pop: np.ndarray, idx:np.ndarray, b_idx:np.ndarray, p:np.ndarray, f:np.ndarray):
    """
    Calculates the mutation of a vector based on the
    "current to p-best" mutation. Includes binary crossover. This is
    V_{i, G} = X_{i, G} + F * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
    """
    br = _get_br(ci, b_idx, p)
    for i in crr:
        scale = _get_f(ci, f)
        m_pop_vec[i]=pop[ci,i] + scale * (pop[br,i] - pop[ci,i]) + scale * (pop[idx[0],i] - pop[idx[1],i])



# def current_to_rand_1_mutation(population: np.ndarray,
#                               population_fitness: np.ndarray,
#                               k: List[float],
#                               f: List[float],
#                               bounds: np.ndarray) -> np.ndarray:
#     """
#     Calculates the mutation of the entire population based on the
#     "current to rand/1" mutation. This is
#     U_{i, G} = X_{i, G} + K * (X_{r1, G} - X_{i, G} + F * (X_{r2. G} - X_{r3, G}
#     :param population: Population to apply the mutation
#     :type population: np.ndarray
#     :param population_fitness: Fitness of the given population
#     :type population_fitness: np.ndarray
#     :param f: Parameter of control of the mutation. Must be in [0, 2].
#     :type f: Union[int, float]
#     :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
#     :type p: Union[int, float]
#     :param bounds: Numpy array of tuples (min, max).
#                    Each tuple represents a gen of an individual.
#     :type bounds: np.ndarray
#     :rtype: np.ndarray
#     :return: Mutated population
#     """
#     # If there's not enough population we return it without mutating
#     if len(population) <= 3:
#         return population
#
#     # 1. For each number, obtain 3 random integers that are not the number
#     parents = __parents_choice(population, 3)
#     # 2. Apply the formula to each set of parents
#     mutated = k * (population[parents[:, 0]] - population)
#     mutated += f * (population[parents[:, 1]] - population[parents[:, 2]])
#
#     return keep_bounds(mutated, bounds)

@nb.njit(**nb_pcs())
def c_to_rand1_mutate(crr:np.ndarray, ci:np.int64, m_pop_vec: np.ndarray, pop: np.ndarray, idx:np.ndarray, f:np.ndarray, k:np.ndarray):
    """
    Calculates the mutation of a vector based on the
    "current to rand/1" mutation. crr are the crossover indexes, m_pop_vec is already inited with pop.
    U_{i, G} = X_{i, G} + K * (X_{r1, G} - X_{i, G} + F * (X_{r2. G} - X_{r3, G}
    """
    for i in crr:
        m_pop_vec[i]=pop[ci,i]+ k[ci] * (pop[idx[0],i] - pop[ci,i]) + f[ci] * (pop[idx[1],i] - pop[idx[2],i])


# def current_to_pbest_weighted_mutation(population: np.ndarray,
#                                        population_fitness: np.ndarray,
#                                        f: np.ndarray,
#                                        f_w: np.ndarray,
#                                        p: float,
#                                        bounds: np.ndarray) -> np.ndarray:
#     """
#     Calculates the mutation of the entire population based on the
#     "current to p-best weighted" mutation. This is
#     V_{i, G} = X_{i, G} + F_w * (X_{p_best, G} - X_{i, G} + F * (X_{r1. G} - X_{r2, G}
#     :param population: Population to apply the mutation
#     :type population: np.ndarray
#     :param population_fitness: Fitness of the given population
#     :type population_fitness: np.ndarray
#     :param f: Parameter of control of the mutation. Must be in [0, 2].
#     :type f: np.ndarray
#     :param f_w: NumPy Array with the weighted version of the mutation array
#     :type f_w: np.ndarray
#     :param p: Percentage of population that can be a p-best. Muest be in (0, 1).
#     :type p: Union[int, float]
#     :param bounds: Numpy array of tuples (min, max).
#                    Each tuple represents a gen of an individual.
#     :type bounds: np.ndarray
#     :rtype: np.ndarray
#     :return: Mutated population
#     """
#     # If there's not enough population we return it without mutating
#     if len(population) < 4:
#         return population
#
#     # 1. We find the best parent
#     best_index = np.argsort(population_fitness)[:max(2, round(p*len(population)))]
#
#     p_best = np.random.choice(best_index, len(population))
#     # 2. We choose two random parents
#     parents = __parents_choice(population, 2)
#     mutated = population + f_w * (population[p_best] - population)
#     mutated += f * (population[parents[:, 0]] - population[parents[:, 1]])
#
#     return keep_bounds(mutated, bounds)

@nb.njit(**nb_pcs())
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
        s1 = _get_f(ci, f)
        s2 = _get_f(ci, f_w)
        m_pop_vec[i]+=s2 * (pop[br,i] - pop[ci,i]) + s1 * (pop[idx[0],i] - pop[idx[1],i])


_N=types.none


def enforce_bounds(population: np.ndarray,
                   bounds: np.ndarray, enf_bds):pass
@overload(enforce_bounds,inline='always')
def _enforce_bounds(population, bounds,enf_bds):
    if enf_bds is _N:
        def _eb(population, bounds,enf_bds):
            pass
    else:
        def _eb(population, bounds,enf_bds):
            keep_bounds(population,bounds)
    return _eb


@nb.njit(**nb_pcs())
def bin_cross_init(m_pop_vec:np.ndarray,pop_vec:np.ndarray,idxg:np.ndarray,cr): #I'll assume cr is still an int for now. add array overload if it is happened on.
    sh=idxg.shape[0]
    l=0
    j_rand = rand.randint(0, idxg.shape[0] - 1)
    for i in range(0,sh):
        if rand.uniform(0., 1.) < cr or i == j_rand:
            idxg[l]=i
            l+=1
        else:
            m_pop_vec[i]=pop_vec[i]
    return idxg[:l]


#uchoice_mutator : unique choice mutator, incorporates the current-member-skipping choice selection
@nb.njit(**nb_cs())
def uchoice_mutator(population: np.ndarray,
                    m_pop:np.ndarray,
                    cr: float,
                    bounds: np.ndarray,
                    enf_bounds:bool,  #None or anything, None will exclude compilation with search bounds enforcement, theres no reason to have anything but None regardless.
                    reject_mx:int, #None or int, None will use the smaller no-resampling compilation.
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
    #To get even more of a boost it might be worth sorting current vectors by proximity to boundaries and equally spreading that load.
    #As closer to boundary params will get more rejections.

    #if pop_size*pop_dims*num_resamples< 100k to 500k then probably not worth parallel, the overhead of launching parallel threads can be like 5-10x a couple hundred ops..
    _uchoice_loop(population, m_pop, bounds, reject_mx, cr, cross_apply, mut_apply, _ns, _idx, _crossgen, _t_pop, *mut_args)
    #print('Average # of failed resample dimensions per vector: ', rsmps.sum()/(pop_size*population.shape[1]))
    enforce_bounds(m_pop,bounds,enf_bounds)


def _uchoice_loop(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args): pass
@overload(_uchoice_loop)
def _uchoice_loop_(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args):
    if reject_mx is _N or bounds is _N or _t_pop is _N:
        def _r(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args):
            _uchoice_nsample_loop(population,m_pop,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,*mut_args)
        return _r
    else:
        def _r(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args):
            _uchoice_rejectsample_loop(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args)
        return _r


@nb.njit(**nb_cs())
def _uchoice_rejectsample_loop(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args):
    pop_size = population.shape[0]
    p_d=population.shape[1]
    reps=reject_mx #min(reject_mx, max(pcfg.reject_sample_mn, ceil(pcfg.reject_sample_mult * (p_d ** pcfg.reject_sample_dimorder))))
    # if pop_size*pop_dims*num_resamples< 100k to 500k then probably not worth parallel, the overhead of launching parallel threads can be like 5-10x a couple hundred ops..
    if pop_size*(p_d+20)*reject_mx//8>1100: #If you are being really anal estimate the # avg of rejections from the previous iterations with an extra array for the divisor.
        nthds=nb.get_num_threads()
        ld=nb.set_parallel_chunksize(ceil(population.shape[0]/nthds)) #consider making a chunk size merging system when total # ops is clearly small enough.
        for v in nb.prange(0, pop_size):
            tid=nb.get_thread_id()
            #hoping these don't cost anything, so it will pick up on it's last memory location.
            _s_mem=np.empty((_ns,2),dtype=np.int64) #Yeah does seem a little quicker
            _bb=p_d+1
            _ccr=cross_apply(_t_pop[tid],population[v],_crossgen[tid],cr)
            for _ in range(reps): #This might get more likely to hit with more dimensions so consider a scaling thing, maybe dynamic.
                _sw_in(_ns, pop_size, v, _idx, tid, _s_mem)
                mut_apply(_ccr,v,_t_pop[tid],population,_idx[tid,:_ns],*mut_args)
                _sw_out(_ns, _idx, tid, _s_mem)
                sb=bounds_safe(_t_pop[tid],bounds)
                if sb<_bb:
                    m_pop[v]=_t_pop[tid] #This and bounds safe are probably what is contributing to the linear delay scaling with # dims.
                    _bb=sb
                    if sb==0:
                        break
        nb.set_parallel_chunksize(ld)
    else:
        for v in range(0, pop_size):
            _s_mem = np.empty((_ns, 2), dtype=np.int64)
            _bb = p_d + 1
            _ccr = cross_apply(_t_pop[0], population[v], _crossgen[0], cr)
            for _ in range(reps):
                _sw_in(_ns, pop_size, v, _idx, 0, _s_mem)
                mut_apply(_ccr, v, _t_pop[0], population, _idx[0, :_ns], *mut_args)
                _sw_out(_ns, _idx, 0, _s_mem)
                sb = bounds_safe(_t_pop[0], bounds)
                if sb < _bb:
                    m_pop[v] = _t_pop[0]
                    _bb = sb
                    if sb == 0:
                        break


@nb.njit(**nb_cs())
def _uchoice_nsample_loop(population,m_pop,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,*mut_args):
    pop_size = population.shape[0]
    p_d=population.shape[1]
    if pop_size*(p_d+20)>1100: #could end up being different for different computers. For high pop count it's still more efficient. hense the +8.
        nthds=nb.get_num_threads()
        ld=nb.set_parallel_chunksize(ceil(population.shape[0]/nthds)) #consider making a chunk size merging system when total # ops is clearly small enough.
        for v in nb.prange(0, pop_size):
            tid=nb.get_thread_id()
            _s_mem=np.empty((_ns,2),dtype=np.int64) #Yeah does seem a little quicker
            _bb=p_d+1
            _ccr=cross_apply(m_pop[v],population[v],_crossgen[tid],cr)
            _sw_in(_ns, pop_size, v, _idx, tid, _s_mem)
            mut_apply(_ccr,v,m_pop[v],population,_idx[tid,:_ns],*mut_args)
            _sw_out(_ns, _idx, tid, _s_mem)
        nb.set_parallel_chunksize(ld)
    else:
        for v in range(0, pop_size):
            _s_mem = np.empty((_ns, 2), dtype=np.int64)
            _ccr = cross_apply(m_pop[v], population[v], _crossgen[0], cr)
            _sw_in(_ns, pop_size, v, _idx, 0, _s_mem)
            mut_apply(_ccr, v, m_pop[v], population, _idx[0, :_ns], *mut_args)
            _sw_out(_ns, _idx, 0, _s_mem)


#@nb.njit(**nb_cs())
@nb.njit(inline='always')
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

@nb.njit(inline='always')
def _sw_out(n_shuffles, _idx, tid, _s_mem):
    for i in range(n_shuffles - 1, -1, -1):
        x, y = _s_mem[i, 0], _s_mem[i, 1]
        tmp = _idx[tid, x]
        _idx[tid, x] = _idx[tid, y]
        _idx[tid, y] = tmp

# def crossover(population: np.ndarray, mutated: np.ndarray,
#               cr: Union[int, float]) -> np.ndarray:
#     """
#     Crosses gens from individuals of the last generation and the mutated ones
#     based on the crossover rate. Binary crossover
#     :param population: Previous generation population.
#     :type population: np.ndarray
#     :param mutated: Mutated population.
#     :type population: np.ndarray
#     :param cr: Crossover rate. Must be in [0,1].
#     :type population: Union[int, float]
#     :rtype: np.ndarray
#     :return: Current generation population.
#     """
#     chosen = np.random.rand(*population.shape)
#     j_rand = np.random.randint(0, population.shape[1])
#     chosen[j_rand::population.shape[1]] = 0
#     return np.where(chosen <= cr, mutated, population)
#
# def exponential_crossover(population: np.ndarray, mutated: np.ndarray,
#                           cr: Union[int, float]) -> np.ndarray:
#     """
#         Crosses gens from individuals of the last generation and the mutated ones
#         based on the crossover rate. Exponential crossover.
#         :param population: Previous generation population.
#         :type population: np.ndarray
#         :param mutated: Mutated population.
#         :type population: np.ndarray
#         :param cr: Crossover rate. Must be in [0,1].
#         :type population: Union[int, float]
#         :rtype: np.ndarray
#         :return: Current generation population.
#     """
#     if type(cr) is int or float:
#         cr = np.array([cr] * len(population))
#     else:
#         cr = cr.flatten()
#
#     def __exponential_crossover_1(x: np.ndarray, y: np.ndarray, cr: Union[int, float]) -> np.ndarray:
#         z = x.copy()
#         n = len(x)
#         k = np.random.randint(0, n)
#         j = k
#         l = 0
#         while True:
#             z[j] = y[j]
#             j = (j + 1) % n
#             l += 1
#             if np.random.randn() >= cr or l == n:
#                 return z
#
#     return np.array([__exponential_crossover_1(population[i], mutated[i], cr.flatten()[i]) for i in range(len(population))])

@nb.njit(**nb_pcs())
def _selection(population: np.ndarray, new_population: np.ndarray,
              fitness: np.ndarray, new_fitness: np.ndarray,
              return_indexes: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Selects the best individuals based on their fitness.
    :param population: Last generation population.
    :type population: np.ndarray
    :param new_population: Current generation population.
    :type new_population: np.ndarray
    :param fitness: Last generation fitness.
    :type fitness: np.ndarray
    :param new_fitness: Current generation fitness
    :param return_indexes: When active the function also returns the individual indexes that have been modified
    :type return_indexes: bool
    :rtype: ndarray
    :return: The selection of the best of previous generation
     and mutated individual for the entire population and optionally, the indexes changed
    """
    indexes = np.where(fitness > new_fitness)[0]
    population[indexes] = new_population[indexes]
    if return_indexes:
        return population, indexes
    else:
        return population


@nb.njit(**nb_pcs())
def select_better(fitness: np.ndarray, new_fitness: np.ndarray,_idxs: np.ndarray) -> np.ndarray:
    """
    Selects the best individuals based on their fitness.
    """
    bt=0
    for i in range(fitness.shape[0]):
        if fitness[i]>new_fitness[i]:
            _idxs[bt]=i
        bt+=1
    return _idxs[:bt]

