import numpy as np
import numba as nb
from scipy.stats._qmc import Sobol
import gopt.config as pcfg
#from gopt.config import reject_sample_mx,reject_sample_dimorder,reject_sample_mult,reject_sample_min,force_search_bounds,BIN_MUTATE_R
from typing import Callable, Union, List, Tuple, Any
import random as rand
from math import ceil,floor
from numba import njit, types
from numba.core.extending import overload


numba_comp_settings=dict(fastmath=True,parallel=True,error_model='numpy',cache=True)

def nb_cs():
    return numba_comp_settings

def nb_pcs():
    d=numba_comp_settings.copy()
    d.pop('parallel')
    return d

def _pinit_randuniform(population_size, individual_size,bounds):
    minimum = bounds[:,0]
    maximum = bounds[:,1]
    population = np.random.rand(population_size, individual_size) * (maximum-minimum) + minimum
    return population

def _pinit_soboluniform(population_size, individual_size,bounds):
    sob_engine=Sobol(individual_size) #user should understand base2 for sobol on their own.
    minimum = bounds[:,0]
    maximum = bounds[:,1]
    population = sob_engine.random(population_size) * (maximum-minimum) + minimum
    return population

#@nb.njit(fastmath=True,)
def keep_bounds(population: np.ndarray,bounds: np.ndarray) -> np.ndarray: return np.clip(population, bounds[:,0], bounds[:,1],out=population)


@nb.njit(**nb_pcs())
def _keep_bounds(population: np.ndarray,bounds: np.ndarray):
    #Assuming not worth parallel for this.
    #Would need a huge pop and dim number for this to be worth parallel, and probably means don't use DE.
    ps,pd=population.shape[0],population.shape[1]
    #ol=nb.set_parallel_chunksize(ceil((ps*pd)/nb.get_num_threads()))
    #for v in nb.prange(ps*pd):
    for v in range(ps * pd):
        i=v//pd
        n=v%pd
        #have a feeling max min will be faster even though gauranteed assignment but see.
        population[i,n]=min(max(population[i,n],bounds[n,0]),bounds[n,1])
    #nb.set_parallel_chunksize(ol)
    return population

@nb.njit(**nb_pcs())
def bounds_safe(p_vec: np.ndarray,bounds: np.ndarray):
    sb=np.int64(0)
    for v in range(p_vec.shape[0]):
        if not (bounds[v,0]<=p_vec[v]<=bounds[v,1]):
            sb+=1
    #counts how many parameters are within bounds, as there is no other preference for infeasibles besides falling within constraints.
    return sb



def init_population(population_size: int,
                    bounds: np.ndarray,rand_spec: Callable|str|np.ndarray='sobol') -> np.ndarray:
    """
    Creates a random population within its constrained bounds.
    :param population_size: Number of individuals desired in the population.
    :type population_size: int
    :param individual_size: Number of features/gens.
    :type individual_size: int
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: Union[np.ndarray, list]
    :rtype: np.ndarray
    :return: Initialized population.
    """
    individual_size = bounds.shape[0]
    if rand_spec=='sobol':
        _p=_pinit_soboluniform(population_size,individual_size,bounds)
    elif rand_spec=='uniform':
        _p = _pinit_randuniform(population_size, individual_size, bounds)
    elif isinstance(rand_spec,Callable):
        _p=rand_spec(population_size, individual_size, bounds)
    else: #assume it's initialized numpy array.
        _p=rand_spec
    keep_bounds(_p,bounds)

    return _p

def __parents_choice(population: np.ndarray, n_parents: int) -> np.ndarray:
    #Testing to see how much overhead can be conveniently reduced.
    pob_size = population.shape[0]
    choices = np.indices((pob_size, pob_size))[1]
    #print('csi',np.indices((pob_size, pob_size)))
    #print('csi1',np.indices((pob_size,)))
    #print('cs',choices)
    mask = np.ones(choices.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    #print('mk',mask)
    choices = choices[mask].reshape(pob_size, pob_size - 1)
    #print('cs2',choices)
    parents = np.array([np.random.choice(row, n_parents, replace=False) for row in choices])
    #print('ps',parents)

    return parents

#This now needs to be merged with the mutation strategy so that we can select bounds or feasible rejection sampling.
#Shouldn't need to worry about duplicate mutations, as the crossover should make identical proposal probability very unlikely.
@nb.njit(fastmath=True)
def __parents_choice_new(p_choice: np.ndarray[np.int64]) -> np.ndarray:
    pop_size=p_choice.shape[0]
    s_size=p_choice.shape[1] #is always at least pop_size-1
    #n_ex=0
    for s in range(0,pop_size):
        #select [0,pop_size-1] int, offset by +1 starting at s, without replacement into p_choice[s]
        #...

        #n_ex+=1
        pass


#@nb.njit
def durstenfeld_partial_shuffle(a, k):
    """
    Perform up to k swaps of the Durstenfeld shuffle on array 'a',
    storing each swap in 'swap_memory' and then reversing them
    to restore 'a' to its original order.
    Given
    """
    n = a.shape[0]
    num_swaps = min(k, n - 1)
    swap_memory = np.zeros((num_swaps, 2), dtype=np.int64)

    # Forward pass: shuffle (up to k swaps)
    for i in range(num_swaps):
        # Generate a random index j with i <= j < n
        j = rand.randrange(i,n)
        # Record swapped indices
        swap_memory[i, 0] = i
        swap_memory[i, 1] = j
        # Swap in-place
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp

    # Backward pass: revert the swaps
    for i in range(num_swaps - 1, -1, -1):
        x, y = swap_memory[i, 0], swap_memory[i, 1]
        tmp = a[x]
        a[x] = a[y]
        a[y] = tmp

    return swap_memory



#@nb.njit(fastmath=True)
def binary_mutation(population: np.ndarray,
                    m_pop:np.ndarray,
                    f: np.float64,
                    bounds: np.ndarray,
                    #_idx
                    ) -> np.ndarray:
    #binary mutation, binary crossover.
    s_size=3
    if len(population) <= 3:
        return population
    parents = __parents_choice(population, 3)
    m_pop[:] = f * (population[parents[:, 0]] - population[parents[:, 1]]) + population[parents[:, 2]]
    #mutated += population[parents[:, 2]]

    return keep_bounds(m_pop, bounds)


@nb.njit(**nb_cs())
def _bin_crossover_o(pop_vec: np.ndarray, m_pop_vec: np.ndarray,cr,j_rand) -> np.ndarray:
    #Add compile options for if cr is an array and j_rand is an array.
    j_rand = rand.randint(0, pop_vec.shape[0])
    for i in range(pop_vec.shape[0]):
        if rand.uniform(0.,1.)>cr or i!=j_rand:
            pass


#So current template is:
#ci:np.int64, m_pop: np.ndarray,pop: np.ndarray,idx:np.ndarray, *args
#ci is the current index, because of crossovers it is always utilized.
#
#*args : best idxs, cr, crw, f, fw, h etc go here.
@nb.njit(**nb_pcs())
def bin_mutate_bin_cross_o(ci:np.int64,m_pop_vec: np.ndarray,pop: np.ndarray,idx:np.ndarray,f,cr,j_rand):
    for i in range(m_pop_vec.shape[0]):
        m_pop_vec[i]=f * (pop[idx[0],i] - pop[idx[1],i]) + pop[idx[2],i] if rand.uniform(0., 1.) < cr or i == j_rand else pop[ci,i]


#uchoice_mutator : unique choice mutator, incorporates the current idx skipping choice selection
@nb.njit(**nb_cs())
def uchoice_mutator_o(population: np.ndarray,
                    m_pop:np.ndarray,
                    bounds: np.ndarray,
                    n_shuffles:np.int64,
                    mut_apply,
                    _idx: np.ndarray, #should already be initialized (thread count, len_pop) with ints [0,n) for thread parallelism.
                    _t_pop:np.ndarray, #temporary population holder, needed to replace the best infeasible.
                    *args): #best idxs, cr, crw, f, fw, h etc go here for dynamic compile.
    pop_size = population.shape[0]
    p_d=population.shape[1]
    reps=min(pcfg.reject_sample_mx, max(pcfg.reject_sample_mn, ceil(pcfg.reject_sample_mult * (p_d ** pcfg.reject_sample_dimorder))))
    nthds=nb.get_num_threads()
    # rsmps=np.zeros((nthds,),dtype=np.int64)
    ld=nb.set_parallel_chunksize(ceil(pop_size/nthds)) #consider making a chunk size merging system when total # ops is clearly small enough.
    for v in nb.prange(0, pop_size):
        tid=nb.get_thread_id()
        #hoping these don't cost anything, so it will pick up on it's last memory location.
        _s_mem=np.empty((n_shuffles,2),dtype=np.int64) #Yeah does seem a bit quicker
        #_t_pop=np.empty((p_d,),dtype=np.float64) #Not this tho
        _bb=population.shape[1]+1
        #Rejection sampling will now be a "closest to feasible" sampler to diminish bias, but becomes too expensive the longer it goes.
        #So the user still needs to enforce actually infeasible bounds in the parameter space, so that optimizer is unconstrained in the search space.
        #Then add a fitness discount to eval function. Or enable them in the search space in config. Though some optimizers like lshade-cnepsin don't respect them.
        for _ in range(reps): #This might get more likely to hit with more dimensions so consider a scaling thing, maybe dynamic.
            for i in range(n_shuffles):
                # Generate a random index j with i <= j < n
                j = rand.randint(i, pop_size - 2) #we only skip j, i gives us our selection
                j= j+1 if j>=v else j #that skip current vector thingy.
                # Record swapped indices
                _s_mem[i, 0] = i
                _s_mem[i, 1] = j
                # Swap in-place
                tmp = _idx[tid,i]
                _idx[tid,i] = _idx[tid,j]
                _idx[tid,j] = tmp
            mut_apply(v,_t_pop[tid],population,_idx[tid,:n_shuffles],*args)
            # Backward pass: revert the random swaps
            for i in range(n_shuffles - 1, -1, -1):
                x, y = _s_mem[i, 0], _s_mem[i, 1]
                tmp = _idx[tid,x]
                _idx[tid,x] = _idx[tid,y]
                _idx[tid,y] = tmp
            sb=bounds_safe(_t_pop[tid],bounds)
            if sb<_bb:
                m_pop[v]=_t_pop[tid] #This and bounds safe are probably what is contributing to the linear delay scaling with # dims.
                _bb=sb
                if sb==0:
                    break
    #     else:
    #         rsmps[tid]+=_bb
    nb.set_parallel_chunksize(ld)
    #print('Average # of failed resample dimensions per vector: ', rsmps.sum()/(pop_size*population.shape[1]))
    if pcfg.force_search_bounds:
        _keep_bounds(m_pop,bounds) #force search bounds.


def crossover_c(population: np.ndarray, mutated: np.ndarray,
              cr: Union[int, float]) -> np.ndarray:
    chosen = np.random.rand(*population.shape)
    j_rand = np.random.randint(0, population.shape[1])
    chosen[j_rand::population.shape[1]] = 0
    print('chosen\n',chosen)
    return np.where(chosen <= cr, mutated, population)


def crossover_o(population: np.ndarray, mutated: np.ndarray,
              cr: Union[int, float]) -> np.ndarray:
    chosen = np.random.rand(*population.shape)
    j_rand = np.random.randint(0, population.shape[1])
    chosen[:,j_rand] = 0
    print('chosen\n',chosen)
    return np.where(chosen <= cr, mutated, population)

def crossover(population: np.ndarray, mutated: np.ndarray,
              cr: Union[int, float]) -> np.ndarray:
    chosen = np.random.rand(*population.shape)
    j_rand = np.random.randint(0, population.shape[1],size=population.shape[0])
    print(j_rand)
    chosen[np.arange(population.shape[0]),j_rand] = 0
    #chosen[:,j_rand] = 0
    print('chosen\n',chosen)
    return np.where(chosen <= cr, mutated, population)


def exponential_crossover(population: np.ndarray, mutated: np.ndarray,
                          cr: Union[int, float]) -> np.ndarray:
    if type(cr) is int or float:
        cr = np.array([cr] * len(population))
    else:
        cr = cr.flatten()

    def __exponential_crossover_1(x: np.ndarray, y: np.ndarray, cr: Union[int, float]) -> np.ndarray:
        z = x.copy()
        n = len(x)
        k = np.random.randint(0, n)
        j = k
        l = 0
        while True:
            z[j] = y[j]
            j = (j + 1) % n
            l += 1
            if np.random.randn() >= cr or l == n:
                return z

    return np.array([__exponential_crossover_1(population[i], mutated[i], cr.flatten()[i]) for i in range(len(population))])

# Old Above
_N=types.none

@nb.njit(**nb_pcs())
def bin_mutate(crr:np.ndarray, ci:np.int64, m_pop: np.ndarray, pop: np.ndarray, idx:np.ndarray, f):
    for i in crr:
        m_pop[i]=f * (pop[idx[0],i] - pop[idx[1],i]) + pop[idx[2],i]


def enforce_bounds(population: np.ndarray,bounds: np.ndarray, enf_bds):pass
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
                    enf_bounds:bool,  #None or anything
                    reject_mx:int,
                    cross_apply,  #these should probably have underscores... but lets user implement their own.
                    mut_apply,
                    _ns: np.int64,  #number of r selections for mutation operator
                    _idx: np.ndarray,  #should already be initialized (thread count, len_pop) with ints [0,n) for thread parallelism.
                    _crossgen:np.ndarray,
                    _t_pop:np.ndarray,  #temporary population holder, needed to replace the best infeasible.
                    *mut_args): #best idxs, cr, crw, f, fw, h etc go here for dynamic compile.
    #Rejection sampling is a "closest to feasible" sampler to diminish bias, but the cost is ~ O(C*pop_size*pop_dims*num_resamples), so can become expensive.
    #So the user still needs to enforce actually infeasible bounds in the parameter space, so that optimizer is unconstrained in the search space. (unless they enforce_bounds in search space, not recommended)
    #Then add a fitness discount to eval function. Or enable them in the search space in config. Though some optimizers like lshade-cnepsin don't respect them.
    #To get even more of a boost it might be worth sorting current vectors by proximity to boundaries and equally spreading that load.
    #As closer to boundary params will get more rejections.
# uchoice_mutator(population, m_pop, cr, bounds, enf_bounds, reject_mx, cross_apply, mut_apply, _ns, _idx, _crossgen, _t_pop, *mut_args)

    _uchoice_loop(population, m_pop, bounds, reject_mx, cr, cross_apply, mut_apply, _ns, _idx, _crossgen, _t_pop, *mut_args)
    #print('Average # of failed resample dimensions per vector: ', rsmps.sum()/(pop_size*population.shape[1]))
    enforce_bounds(m_pop,bounds,enf_bounds)

@nb.njit(**nb_pcs())
def uchoice_mutator_s(population: np.ndarray,
                    m_pop:np.ndarray,
                    cr: float,
                    bounds: np.ndarray,
                    enf_bounds:bool,  #None or anything
                    reject_mx:int,
                    cross_apply,  #these should probably have underscores... but lets user implement their own.
                    mut_apply,
                    _ns: np.int64,  #number of r selections for mutation operator
                    _idx: np.ndarray,  #should already be initialized (thread count, len_pop) with ints [0,n) for thread parallelism.
                    _crossgen:np.ndarray,
                    _t_pop:np.ndarray,  #temporary population holder, needed to replace the best infeasible.
                    *mut_args): #best idxs, cr, crw, f, fw, h etc go here for dynamic compile.
    #Rejection sampling is a "closest to feasible" sampler to diminish bias, but the cost is ~ O(C*pop_size*pop_dims*num_resamples), so can become expensive.
    #So the user still needs to enforce actually infeasible bounds in the parameter space, so that optimizer is unconstrained in the search space. (unless they enforce_bounds in search space, not recommended)
    #Then add a fitness discount to eval function. Or enable them in the search space in config. Though some optimizers like lshade-cnepsin don't respect them.
    #To get even more of a boost it might be worth sorting current vectors by proximity to boundaries and equally spreading that load.
    #As closer to boundary params will get more rejections.

    _uchoice_loop_s(population, m_pop, bounds, reject_mx, cr, cross_apply, mut_apply, _ns, _idx, _crossgen, _t_pop, *mut_args)
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

def _uchoice_loop_s(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args): pass
@overload(_uchoice_loop_s)
def _uchoice_loop_s_(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args):
    if reject_mx is _N or bounds is _N or _t_pop is _N:
        def _r(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args):
            _uchoice_nsample_loop_s(population,m_pop,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,*mut_args)
        return _r
    else:
        def _r(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args):
            _uchoice_rejectsample_loop_s(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args)
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


@nb.njit(**nb_pcs())
def _uchoice_rejectsample_loop_s(population,m_pop,bounds,reject_mx,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,_t_pop,*mut_args):
    pop_size = population.shape[0]
    p_d=population.shape[1]
    reps=reject_mx #min(reject_mx, max(pcfg.reject_sample_mn, ceil(pcfg.reject_sample_mult * (p_d ** pcfg.reject_sample_dimorder))))
    for v in range(0, pop_size):
        _s_mem=np.empty((_ns,2),dtype=np.int64)
        _bb=p_d+1
        _ccr=cross_apply(_t_pop[0],population[v],_crossgen[0],cr)
        for _ in range(reps):
            _sw_in(_ns, pop_size, v, _idx, 0, _s_mem)
            mut_apply(_ccr,v,_t_pop[0],population,_idx[0,:_ns],*mut_args)
            _sw_out(_ns, _idx, 0, _s_mem)
            sb=bounds_safe(_t_pop[0],bounds)
            if sb<_bb:
                m_pop[v]=_t_pop[0]
                _bb=sb
                if sb==0:
                    break


@nb.njit(**nb_cs())
def _uchoice_nsample_loop(population,m_pop,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,*mut_args):
    pop_size = population.shape[0]
    p_d=population.shape[1]
    if pop_size*(p_d+20)>1100: #could end up being different for different computers.
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
            _bb = p_d + 1
            _ccr = cross_apply(m_pop[v], population[v], _crossgen[0], cr)
            _sw_in(_ns, pop_size, v, _idx, 0, _s_mem)
            mut_apply(_ccr, v, m_pop[v], population, _idx[0, :_ns], *mut_args)
            _sw_out(_ns, _idx, 0, _s_mem)



@nb.njit(**nb_pcs())
def _uchoice_nsample_loop_s(population,m_pop,cr,cross_apply,mut_apply, _ns, _idx,_crossgen,*mut_args):
    pop_size = population.shape[0]
    p_d=population.shape[1]
    for v in range(0, pop_size):
        _s_mem=np.empty((_ns,2),dtype=np.int64)
        _bb=p_d+1
        _ccr=cross_apply(m_pop[v],population[v],_crossgen[0],cr)
        _sw_in(_ns, pop_size, v, _idx, 0, _s_mem)
        mut_apply(_ccr,v,m_pop[v],population,_idx[0,:_ns],*mut_args)
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

import time as t
import numpy as np
import time
import matplotlib.pyplot as plt

def parallel_tradeoff(n_init,test_sync, test_pl, repeats=10000, n_start=10, n_end=100, ext_repeats=5):
    t.sleep(1.) #let the interpreter calm down
    args = n_init(n_end)
    test_sync(10, *args)
    test_pl(10, *args)
    t.sleep(1.)
    n_vals, sync_vals, parallel_vals = [], [], []
    for n in range(n_end,n_start-1,-1):
        sync_t, par_t = 0.0, 0.0
        args = n_init(n)
        for _ in range(ext_repeats):
            t0 = time.perf_counter(); test_sync(repeats, *args); t1 = time.perf_counter()
            sync_t += (t1 - t0)
            t0 = time.perf_counter(); test_pl(repeats, *args); t1 = time.perf_counter()
            par_t += (t1 - t0)
        n_vals.append(n)
        sync_vals.append(sync_t*1000 / (ext_repeats*repeats))
        parallel_vals.append(par_t*1000 / (ext_repeats*repeats))
        print(f'Completed n: {n}, average sync ms:{sync_vals[-1]:.5f}, average parallel ms:{parallel_vals[-1]:.5f}')
    return np.array(n_vals), np.array(sync_vals), np.array(parallel_vals)

def plot_tradeoff(n_vals, sync_vals, parallel_vals,log=False):
    plt.figure(figsize=(16, 10))
    plt.plot(n_vals, np.log2(sync_vals) if log else sync_vals, label='Synchronous', color='blue')
    plt.plot(n_vals, np.log2(parallel_vals) if log else parallel_vals, label='Parallel', color='red')
    plt.xlabel('n-value'), plt.ylabel('Mean Time (seconds)')
    plt.title('Performance Tradeoff: Synchronous vs Parallel')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(), plt.tight_layout(), plt.show()

def _ini_pt(dim,pop_sz=1024):
    bds = np.empty((dim, 2), dtype=np.float64)
    bds[:, 0] = 1
    bds[:, 1] = 10
    p_ = init_population(pop_sz, bds)
    bds[:, 0] = 2.
    bds[:, 1] = 9.  # so basically, init it where you think it's likely, add a buffer..
    return p_,bds

def _ini_h(population:np.ndarray):
    nt=nb.get_num_threads()
    m_pop=np.empty(population.shape,dtype=np.float64)
    _crr = np.empty((nt,population.shape[1]),dtype=np.int64)
    _t_pop=np.empty((nt,population.shape[1]),dtype=np.float64)
    _idx = np.empty((nt, population.shape[0]), dtype=np.int64)
    _idx[:]= np.arange(0,population.shape[0])
    return m_pop,_crr, _t_pop,_idx

@nb.njit(**nb_cs())
def test_mutator_pl(repeats,
                   population: np.ndarray,
                    m_pop:np.ndarray,
                    cr: float,
                    bounds: np.ndarray,
                    enf_bounds:bool, #None or anything, None will exclude compilation with search bounds enforcement, theres no reason to have anything but None regardless.
                    reject_mx:int, #None or int, None will use the smaller no-resampling compilation.
                    cross_apply,  #these should probably have underscores... but lets user implement their own.
                    mut_apply,
                    _ns: np.int64,  #number of r selections for mutation operator
                    _idx: np.ndarray,  #should already be initialized (thread count, len_pop) with ints [0,n) for thread parallelism.
                    _crossgen:np.ndarray,
                    _t_pop:np.ndarray,  #temporary population holder, needed to replace the best infeasible.
                    *mut_args): #using binary mutation so only f for mut_args

    for _ in range(repeats):
        uchoice_mutator(population, m_pop, cr, bounds, enf_bounds, reject_mx, cross_apply, mut_apply, _ns, _idx, _crossgen, _t_pop, *mut_args)

@nb.njit(**nb_pcs())
def test_mutator_s(repeats,
                    population: np.ndarray,
                    m_pop: np.ndarray,
                    cr: float,
                    bounds: np.ndarray,
                    enf_bounds: bool,  # None or anything
                    reject_mx: int,
                    cross_apply,  # these should probably have underscores... but lets user implement their own.
                    mut_apply,
                    _ns: np.int64,  # number of r selections for mutation operator
                    _idx: np.ndarray,
                    # should already be initialized (thread count, len_pop) with ints [0,n) for thread parallelism.
                    _crossgen: np.ndarray,
                    _t_pop: np.ndarray,  # temporary population holder, needed to replace the best infeasible.
                    *mut_args):  # using binary mutation so only f for mut_args

    for _ in range(repeats):
        uchoice_mutator_s(population, m_pop, cr, bounds, enf_bounds, reject_mx, cross_apply, mut_apply, _ns, _idx, _crossgen,
                        _t_pop, *mut_args)


def test_mutate_init(n,pop_sz=1024):
    pop,bounds = _ini_pt(n,pop_sz=pop_sz)
    m_pop,_crr,_t_pop,_idx = _ini_h(pop)
    #see test_mutator for arg orderings
    return pop,m_pop,.5,bounds,None,None,bin_cross_init,bin_mutate,3,_idx,_crr,_t_pop, .5


def mutation_parallel_tradeoff_dim(d_s,d_e,pop_sz,repeats,ext_repeats):
    def tmi(n): return test_mutate_init(n,pop_sz)
    n_vals,sync_vals,parallel_vals=parallel_tradeoff(tmi,test_mutator_s,test_mutator_pl,repeats,d_s,d_e,ext_repeats)
    plot_tradeoff(n_vals,sync_vals, parallel_vals,log=False)


if __name__=='__main__':
    np.set_printoptions(suppress=True)

    #old_run()
    mutation_parallel_tradeoff_dim(10,50,8,80,80)


def old_run():
    dim = 6
    ps = 8
    np.set_printoptions(suppress=True)
    bds = np.empty((dim, 2), dtype=np.float64)
    bds[:, 0] = 1
    bds[:, 1] = 10
    p_ = init_population(ps, bds)
    np_ = np.empty_like(p_, dtype=np.float64)
    bds[:, 0] = .9
    bds[:, 1] = 10.1  # so basically, init it where you think it's likely, add a buffer..
    # np.random.seed(42)
    np_ = binary_mutation(p_, np_, .5, bds)
    # print('og vecs\n',p_)
    # print('first mutate vecs\n', np_)
    # #np.random.seed(42)
    cp = crossover(p_, np_, .5)
    # print('first bin cross vecs\n', cp)
    # print('bin diff\n', cp-p_)
    # print(np.sum(np.abs(cp-p_),axis=0))
    # print(np.sum(np.abs(cp-p_)))
    # np.random.seed(42)
    ph = np.empty((nb.get_num_threads(), ps), dtype=np.int64)
    ph[:] = np.array([i for i in range(0, ps)])
    _t_pop = np.empty((nb.get_num_threads(), dim), dtype=np.float64)
    # _s_mem=np.empty((nb.get_num_threads(),pcfg.BIN_MUTATE_R,2),dtype=np.int64)
    # _s_mem = np.empty((nb.get_num_threads(), BIN_MUTATE_R, 2), dtype=np.int64)
    # print(_t_pop)

    # uchoice_mutator(p_, np_, bds, BIN_MUTATE_R, bin_mutate_bin_cross, ph,.5, .5, rand.randint(0, dim - 1))
    uchoice_mutator(p_, np_, bds, pcfg._BIN_M_R, bin_mutate, ph, _t_pop, .5, .9, rand.randint(0, dim - 1))

    st = t.perf_counter()
    # uchoice_mutator(p_, np_, bds,BIN_MUTATE_R,bin_mutate_bin_cross, ph,.5, .5, rand.randint(0, dim - 1))
    uchoice_mutator(p_, np_, bds, pcfg._BIN_M_R, bin_mutate, ph, _t_pop, .5, .9, rand.randint(0, dim - 1))
    print('one step time:', t.perf_counter() - st)
    # print('first bin resamp cross vecs\n', np_)
    # print('resamp diff\n', np_-p_)
    # print(np.sum(np.abs(np_-p_)[:,],axis=0))
    print(np.sum(np.abs(np_ - p_)[:, ]))
    # print(ph)



