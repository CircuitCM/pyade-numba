import random as rand
import numpy as np
import gopt.commons as cmn
import gopt.config as cfg
from gopt.config import CrossoverSelector,MutationSelector
from typing import Callable, Union, Dict, Any,Sequence
A=np.ndarray
import numba as nb
from numba import njit, types
from numba.core.extending import overload,register_jitable
from math import ceil

_PB_IMP=(MutationSelector.CUR_T_PBEST,)
_MUT_IMP=(MutationSelector.CUR_T_PBEST,MutationSelector.BINARY)


def apply_de(pop_eval: Callable,
             f: float|np.ndarray|None=None,  #If None jitters between .5 and 1.
             cr: float|np.ndarray|None=.9,  #If None jitters between .0 and .95, Original CR value .9 for differential evolution, but this leads to instability kinda often tbh.
             init_spec: Callable | str | np.ndarray = 'sobol',  #If 'sobol', or 'rand' selects from uniform. Otherwise provide array or initializer.
             pop_dim: tuple|list|np.ndarray|None=None,  #If None, then init_spec needs to be a callable or array of (pop_sz, individual_sz).
             bounds: A = None,  #Required if you don't init your own population/supply a callable that uses pop and indiv size.
             max_evals: int = 100000,
             cross_type:int=CrossoverSelector.BIN,  #for now its only bin. can add others if helpful in the future.
             mutation_type:int=MutationSelector.CUR_T_PBEST,
             #enf_bounds: bool = False, #Recommended to keep it False, and truncate parameters to boundaries in your pop_eval. depricated, do it yourself in eval space, not search space.
             reject_sample_max:int= None,  #Must be >1 or None, If None then no rejection_sampling. Can reduce boundary bias, and increase meaningful search in valid boundaries for same # population evals.  ~5 is a good balance when optimal fitness is along boundaries.
             p_best:float|np.ndarray|None=.11,  #Only used if a probability wgt'd mutator is used. jitters between (0,.9] if None.
             seed:int=None,  #42_420_69_9001 fyi seed pretty much doesn't work with multithreading, even though numba should be supporting it.
             stop_condition:Callable|float|None=None,
             monitor_func:Callable|None=None,
             eval_opts:Sequence=(),
             make_callable:bool=False) -> tuple[A | None, A | None] | Callable:
    """
    Applies the standard differential evolution algorithm.
    :param population_size: Size of the population.
    :type population_size: int
    :param individual_size: Number of gens/features of an individual.
    :type individual_size: int
    :param f: Mutation parameter. Must be in [0, 2].
    :type f: Union[float, int]
    :param cr: Crossover Ratio. Must be in [0, 1].
    :type cr: Union[float, int]
    :param bounds: Numpy ndarray with individual_size rows and 2 columns.
    First column represents the minimum value for the row feature.
    Second column represent the maximum value for the row feature.
    :type bounds: A
    :param func: Evaluation function. The function used must receive one
     parameter.This parameter will be a numpy array representing an individual.
    :type func: Callable[[A], float]
    :param opts: Optional parameters for the fitness function.
    :type opts: Any type.
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
    :param cross: Indicates whether to use the binary crossover('bin') or the exponential crossover('exp').
    :type cross: str
    :param max_evals: Number of evaluations after the algorithm is stopped.
    :type max_evals: int
    :param seed: Random number generation seed. Fix a number to reproduce the
    same results in later experiments.
    :type seed: Union[int, None]
    :return: A pair with the best solution found and its fitness.
    :rtype [A, int]
    """

    # if (type(f) is not int and type(f) is not float) or not 0 <= f <= 2:
    #     raise ValueError("f (mutation parameter) must be a "
    #                      "real number in [0,2].")

    # if (type(cr) is not int and type(cr) is not float) or not 0 <= cr <= 1:
    #     raise ValueError("cr (crossover ratio) must be a "
    #                      "real number in [0,1].")

    if type(max_evals) is not int or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

    #bounds can also be none, but init population.
    # if type(bounds) is not A or bounds.shape != (individual_size, 2):
    #     raise ValueError("bounds must be a NumPy ndarray.\n"
    #                      "The array must be of individual_size length. "
    #                      "Each row must have 2 elements.")

    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

    # to avoid recompilations for different jitter params, use numpy arrays instead of tuples (though they are still going to happen with different eval, stop and monitor funcs).
    pp, f, cr, p_best = _ini_non_compiled_settings(f,cr,init_spec,pop_dim,bounds,mutation_type,p_best)

    gus=_ini_h(pp,False if bounds is None or reject_sample_max is None else True)
    max_iters=ceil(max_evals/pp.shape[0]) #a little more than max.
    crf=cmn.bin_cross_init if cross_type == CrossoverSelector.BIN else None #implement later if needed
    stop_apply= cmn.mk_fitness_thresh_stop(stop_condition) if type(stop_condition) is float else stop_condition #leaving none or callable.
    if mutation_type in _MUT_IMP:
        if make_callable:
            #Notes to not have errors: fitness outputs should depend solely on the eval function, all eval opts for pop_eval and monitor
            #need to be managed/cleared by the user after one call, if they retain information of the run.
            print('Return callable selected, attempting jit compilation.\nRequesting a numba compiled callable object may occur in separate python threads to handle many fitness functions.\nHowever optimizers will take advantage of all CPU threads without releasing the GIL to avoid memory inconsistencies in numpy arrays.\nThis means there will be no benefit to running optimizer callables in separate python threads, performance will be best when run sequentially.')
            failed_jit=False
            #get it to compile first
            try:
                run_de(mutation_type,pp, bounds, reject_sample_max, 2, f, p_best, cr, seed, crf,stop_apply, pop_eval,monitor_func, *gus, *eval_opts)
                print('Optimizer successfully compiled.')
            except Exception as e:
                failed_jit=True
                print(f"\n--- Failed to compile callable in nopython mode: {e}\n Attempting python for eval, monitor, and stop functions.")
                run_de_nocompile(mutation_type,pp, bounds, reject_sample_max, 2, f, p_best, cr, seed, crf,stop_apply, pop_eval,monitor_func, *gus, *eval_opts)
            #_run=run_de_nocompile if failed_jit else run_de
            #user is expected to know that changing the types of new callable parameters will cause a new compile.
            def run(n_f: float=None, n_cr: float=None,n_init_spec: Callable | str | np.ndarray = None,
                    n_pop_dim: tuple|list|np.ndarray|None=None, n_bounds: A = None,n_max_evals:int=None,
                    n_reject_sample_max:int=None,n_p_best:float|None=None,n_seed:int=None):
                n_f= f if n_f is None else n_f
                n_cr = cr if n_cr is None else n_cr
                n_init_spec = init_spec if n_init_spec is None else n_init_spec
                n_pop_dim = pop_dim if n_pop_dim is None else n_pop_dim
                n_bounds = bounds if n_bounds is None else n_bounds
                n_p_best = p_best if n_p_best is None else n_p_best
                n_seed = seed if n_seed is None else n_seed
                #print(n_f, n_cr, n_init_spec, n_pop_dim, n_bounds, mutation_type, n_p_best)
                n_pp, n_f, n_cr, n_p_best = _ini_non_compiled_settings(n_f, n_cr, n_init_spec, n_pop_dim, n_bounds, mutation_type, n_p_best)
                n_max_iters = max_iters if n_max_evals is None else ceil(n_max_evals/n_pp.shape[0])
                n_reject_sample_max = reject_sample_max if n_reject_sample_max is None else n_reject_sample_max
                n_gus=gus
                #print(n_f, n_cr, n_init_spec, n_pop_dim, n_bounds, mutation_type, n_p_best)
                if n_pp.shape!=pp.shape:
                    n_gus = _ini_h(n_pp,False if bounds is None or reject_sample_max is None else True)
                if failed_jit:
                    return run_de_nocompile(mutation_type,n_pp,n_bounds,n_reject_sample_max,
                                            n_max_iters,n_f,n_p_best,n_cr,n_seed,crf,stop_apply,pop_eval,monitor_func,*n_gus,*eval_opts)
                else:
                    return run_de(mutation_type, n_pp, n_bounds, n_reject_sample_max, n_max_iters, n_f, n_p_best, n_cr,
                                            n_seed, crf, stop_apply, pop_eval, monitor_func, *n_gus, *eval_opts)

            return run
        else:
            try:
                return run_de(mutation_type,pp, bounds, reject_sample_max, max_iters, f, p_best, cr, seed, crf,stop_apply, pop_eval,monitor_func, *gus, *eval_opts)
            except Exception as e:
                print(f"Failed to run in nopython mode: {e}\n Attempting python for eval, monitor, and stop functions.")
                return run_de_nocompile(mutation_type,pp, bounds, reject_sample_max, max_iters, f, p_best, cr, seed, crf,stop_apply, pop_eval,monitor_func, *gus, *eval_opts)
    else:
        print('Specified mutation strategy is not implemented for this optimizer. Please select an available strategy.')
        return None,None

def _ini_non_compiled_settings(f: float=None, #If None jitters between .5 and 1.
                                cr: float=.9,
                                init_spec: Callable | str | np.ndarray = 'sobol',
                                pop_dim: tuple|list|np.ndarray|None=None,
                                bounds: A = None,
                               mutation_type:int=MutationSelector.CUR_T_PBEST,
                                p_best:float|None=.11,
                               ):
    #These aren't actually all non-compiled functions like when they are None, a numerical, or np array.
    #But they are the ones most likely to tested for benchmarking.
    if pop_dim is not None:
        pp = cmn.init_population(pop_dim[0], pop_dim[1], bounds, init_spec)
    else:
        pp = cmn.init_population(None, None, bounds, init_spec)

    # to avoid recompilations for different jitter params, use numpy arrays instead of tuples (though they are still going to happen with different eval, stop and monitor funcs).

    if mutation_type in _PB_IMP:
        if isinstance(p_best,
                      float):  # Then also assume it's between 0 and 1 no user check... maybe should add if it causes memory leaks instead of crashing though.
            p_best = ceil(pp.shape[0] * p_best)
        elif isinstance(p_best, np.ndarray) and p_best.dtype not in (np.int32,np.int64,np.uint64,np.uint32): #make better type inference later
            p_best = np.array([ceil(pp.shape[0] * p_best[0]), ceil(pp.shape[0] * p_best[1])], dtype=np.int64)
        elif p_best is None:
            pbj = [ceil(pp.shape[0] * 0.01), ceil(pp.shape[0] * .9)] #.4 to .6 might be better tho
            print(f'A probability weighted mutation strategy was selected with no p_best.\nSelecting with jitter range: {pbj}.')
            p_best = np.array(pbj, dtype=np.int64)
    else:
        p_best = None

    if f is None:
        fj = [.25, 1.1] #having it on a wider range than the class .5, 1. seems to do better sometimes.
        print(f'No mutation factor (f) selected. Selecting jitter range: {fj}')
        f = np.array(fj, dtype=np.float64)
    if cr is None:
        cr = [0., .9] #suprisingly jittering cr, by generation can make results much more robust.
        print(f'No crossover rate (cr) selected. Selecting jitter range: {cr}')
        cr = np.array(cr, dtype=np.float64)

    return pp,f,cr,p_best


def _ini_h(population:A,yes=True):
    nt=nb.get_num_threads()
    m_pop=np.empty(population.shape,dtype=np.float64)
    _crr = np.empty((nt,population.shape[1]),dtype=np.int64)
    _t_pop=np.empty((nt,population.shape[1]),dtype=np.float64) if yes else None
    _idx = np.empty((nt+1, population.shape[0]), dtype=np.int64)
    _idx[:-1]= np.arange(0,population.shape[0])
    _ftdiym=np.zeros((population.shape[0],), dtype=np.float64)
    #_adegen = np.empty((nt, population.shape[1]), dtype=np.float64) if yes and anti_dim_degen else None
    return m_pop,_idx[:-1],_crr,_t_pop,_ftdiym,_idx[-1],#_adegen


@nb.njit(**cmn.nb_cs())
def run_de(mutation_selector:int,population: A,
                            bounds: A,
                            reject_mx:int,
                            max_iters:int,
                            f: float|np.ndarray, p:int|np.ndarray, cr: float|np.ndarray,
                            seed: int,
                            cross_apply: Callable,
                            stop_apply: Callable,
                            pop_eval: Callable,
                            monitor:Callable,
                            _m_pop:A,_idx:A,_crr:A,_t_pop:A,_ftdiym:A, _idxs:A,
                            *eval_opts):

    return run_de_nocompile(mutation_selector,population,
                            bounds,
                            reject_mx,
                            max_iters,
                            f, p, cr,
                            seed,
                            cross_apply,
                            stop_apply,
                            pop_eval,
                            monitor,
                            _m_pop,_idx,_crr,_t_pop,_ftdiym, _idxs,
                            *eval_opts)

@register_jitable(**cmn.nb_cs())
def run_de_nocompile(mutation_selector:int,population: A,
                            bounds: A,
                            reject_mx:int,
                            max_iters:int,
                            f: float|np.ndarray, p:int|np.ndarray, cr: float|np.ndarray,
                            seed: int,
                            cross_apply: Callable,
                            stop_apply: Callable,
                            pop_eval: Callable,
                            monitor:Callable,
                            _m_pop:A,_idx:A,_crr:A,_t_pop:A,_ftdiym:A, _idxs:A,
                            *eval_opts):
    #Match case is now implemented in numba so make sure to use the new version.
    match (mutation_selector):
        case cfg._M_BINARY:
            _de_bin_mutate_bc(population,bounds,reject_mx,max_iters,f,cr,seed,cross_apply,stop_apply,pop_eval,monitor,
                                     _m_pop,_idx,_crr,_t_pop,_ftdiym, _idxs,*eval_opts)
        case cfg._M_CUR_T_PBEST:
            _de_c_t_pbest_bc(population,bounds,reject_mx,max_iters,f,p,cr,seed,cross_apply,stop_apply,pop_eval,monitor,
                                    _m_pop,_idx,_crr,_t_pop,_ftdiym, _idxs, *eval_opts)
    #Old implementation, might remove as it might not return the population's best, especially with stochastic fitness functions.
    #User should gather their desired population and fitness info in their monitor function for assurance.
    _b = np.argmin(_ftdiym)
    return population[_b], _ftdiym[_b]

@register_jitable(**cmn.nb_cs())
def _de_bin_mutate_bc(population: A,
                            bounds: A,
                            reject_mx:int,
                            max_iters:int,
                            f: float|np.ndarray, cr: float|np.ndarray,
                            seed: int,
                            cross_apply: Callable,
                            stop_apply:Callable,
                            pop_eval: Callable,
                            monitor: Callable,
                            _m_pop:A,_idx:A,_crr:A,_t_pop:A,_ftdiym:A, _idxs:A,
                            *eval_opts):
    tf,tcr=_first(population,_ftdiym,seed,f,cr,pop_eval,*eval_opts)
    for current_generation in range(max_iters):
        cmn.uchoice_mutator(population, _m_pop, tcr, bounds, reject_mx, cross_apply, cmn.bin_mutate, cfg._BIN_M_R, _idx, _crr, _t_pop, tf)
        tf,tcr=_the_rest(population,_m_pop,_ftdiym,_idxs,f,cr,pop_eval,monitor,*eval_opts)
        if stop_apply is not None and stop_apply(population,_ftdiym):break

@register_jitable(**cmn.nb_cs())
def _de_c_t_pbest_bc(population: A,
                            bounds: A,
                            reject_mx:int,
                            max_iters:int,
                            f: float|np.ndarray, p:int|np.ndarray, cr: float|np.ndarray,
                            seed: int,
                            cross_apply: Callable,
                            stop_apply: Callable,
                            pop_eval: Callable,
                            monitor:Callable,
                            _m_pop:A,_idx:A,_crr:A,_t_pop:A,_ftdiym:A, _idxs:A,
                            *eval_opts):
    tf,tcr=_first(population,_ftdiym,seed,f,cr,pop_eval,*eval_opts)
    #print(tf,tcr)
    _b = np.argsort(_ftdiym) #this will spam new arrays, boohoo
    for current_generation in range(max_iters):
        tp=_pset(p)
        cmn.uchoice_mutator(population, _m_pop, tcr, bounds, reject_mx, cross_apply, cmn.c_to_pbest_mutate, cfg._C_T_PB_M_R, _idx, _crr, _t_pop, _b, tp, tf)
        tf,tcr=_the_rest(population,_m_pop,_ftdiym,_idxs,f,cr,pop_eval,monitor,*eval_opts)
        if stop_apply is not None and stop_apply(population,_ftdiym):break
        _b = np.argsort(_ftdiym)

#Implement pbest2, pbesta, pbest2a later.

@register_jitable(**cmn.nb_pcs())
def _first(population,_ftdiym,seed,f,cr,pop_eval,*eval_opts):
    _rset(seed)
    _ftdiym[:] = pop_eval(population, *eval_opts)  # fitness array sb embedded into eval_opts already. non _'d
    return _fset(f),_fset(cr)

@register_jitable(**cmn.nb_pcs())
def _the_rest(population,m_pop,_ftdiym,_idxs,f,cr,pop_eval,monitor,*eval_opts):
    ftdiym = pop_eval(m_pop, *eval_opts)
    #To capture the most information, monitor comes right before new pop transition.
    _meval(monitor,population,m_pop,_ftdiym,*eval_opts)
    bdx = cmn.select_better(_ftdiym, ftdiym, _idxs)
    _ftdiym[bdx]=ftdiym[bdx]
    population[bdx]=m_pop[bdx]
    return _fset(f),_fset(cr)


@nb.njit(**cmn.nb_pcs())
def _ss(f): rand.seed(f)

def _rset(seed):
    if seed is not None:
        _ss(seed)
        rand.seed(seed)
@overload(_rset,inline='always',**cmn.nb_pcs())
def _rset_(f):
    if isinstance(f,types.NoneType):
        def _f(f): pass
    else:
        def _f(f): return rand.seed(f)
    return _f

#can be used for CR too
def _fset(f):return rand.uniform(f[0], f[1]) if type(f) is np.ndarray else f
@overload(_fset,inline='always',**cmn.nb_pcs())
def _fset_(f):
    if isinstance(f,types.Array):
        def _f(f): return rand.uniform(f[0], f[1])
    else:
        def _f(f): return f
    return _f

def _pset(p):return rand.randrange(p[0], p[1]) if type(p) is np.ndarray else p
@overload(_pset,inline='always',**cmn.nb_pcs())
def _pset_(p):
    if isinstance(p,types.Array):
        def _p(p): return rand.randrange(p[0], p[1]) #jitter p for pbest selection by per generation.
    else:
        def _p(p): return p
    return _p

def _meval(m,*args):
    if m is not None:m(*args)
@overload(_meval,**cmn.nb_cs()) #allowing parallel tho idk what it could end up doing to it if monitor isn't.
def _meval_(m,*args):
    if isinstance(m,types.Callable):
        def _m(m,*args): m(*args)
    else:
        def _m(m,*args): pass
    return _m