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
             make_callable:bool|str=False #False, True, 'for_jitting'
             ) -> tuple[A | None, A | None] | Callable:
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
    #if type(pop_dim) != np.ndarray:
    pop_dim=np.array(pp.shape,dtype=np.int64) # be safe

    stack_mem=init_algo_stack(pp, False if bounds is None or reject_sample_max is None else True)
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
            if make_callable!='for_jitting':
                try:
                    run_de(mutation_type,pp, bounds, reject_sample_max, 2, f, p_best, cr, seed, crf,stop_apply, pop_eval,monitor_func, *stack_mem, *eval_opts)
                    print('Optimizer successfully compiled.')
                except Exception as e:
                    failed_jit=True
                    print(f"\n---WARNING;\nFailed to compile callable in nopython mode: {e}\n Attempting python for eval, monitor, and stop functions.")
                    run_de_nocompile(mutation_type,pp, bounds, reject_sample_max, 2, f, p_best, cr, seed, crf,stop_apply, pop_eval,monitor_func, *stack_mem, *eval_opts)
                _run=run_de_nocompile if failed_jit else run_de
                # user is expected to know that changing the types of new callable parameters will cause a new compile. in _run at the least
                #tunable params are: n_f[0,1],n_cr[0,1],n_cr[0,1],n_pop_dim[0].
                #User can optimize params using this if they don't care about the jit.
                def run(n_f: float = None, n_cr: float = None, n_init_spec: Callable | str | np.ndarray = None,
                        n_pop_dim: np.ndarray | None = None, n_bounds: A = None, n_max_evals: int = None,
                        n_reject_sample_max: int = None, n_p_best: float | None = None, n_seed: int = None):
                    n_f = f if n_f is None else n_f
                    n_cr = cr if n_cr is None else n_cr
                    n_init_spec = init_spec if n_init_spec is None else n_init_spec
                    n_pop_dim = pop_dim if n_pop_dim is None else n_pop_dim
                    n_bounds = bounds if n_bounds is None else n_bounds
                    n_p_best = p_best if n_p_best is None else n_p_best
                    n_seed = seed if n_seed is None else n_seed
                    # print(n_f, n_cr, n_init_spec, n_pop_dim, n_bounds, mutation_type, n_p_best)
                    n_pp = _ini_spc(n_init_spec, n_pop_dim, n_bounds)

                    i0 = n_pop_dim[0] != pop_dim[0]
                    i1 = n_pop_dim[1] != pop_dim[1]
                    n_max_iters = max_iters if n_max_evals is None else ceil(n_max_evals / n_pp.shape[0])
                    n_reject_sample_max = reject_sample_max if n_reject_sample_max is None else n_reject_sample_max
                    n_gus = stack_mem
                    if i0 or i1:
                        n_gus = init_algo_stack(n_pp, False if bounds is None or reject_sample_max is None else True)

                    return _run(mutation_type, n_pp, n_bounds, n_reject_sample_max, n_max_iters, n_f, n_p_best, n_cr,
                                n_seed, crf, stop_apply, pop_eval, monitor_func, *n_gus, *eval_opts)
                return run  # Now it should be callable from a jit function if it compiled successfully, however calling from jit func will trigger second compile.
            else:
                #advanced, user should know what they are doing
                ap=[mutation_type,pp, bounds, reject_sample_max, max_iters, f, p_best, cr, seed, crf,stop_apply, pop_eval,monitor_func]
                ap.extend(stack_mem) #included
                #ap.append(eval_opts) #unecessary as the user should already have this
                return run_de,ap
        else:
            try:
                return run_de(mutation_type,pp, bounds, reject_sample_max, max_iters, f, p_best, cr, seed, crf,stop_apply, pop_eval,monitor_func, *stack_mem, *eval_opts)
            except Exception as e:
                print(f"Failed to run in nopython mode: {e}\n Attempting python for eval, monitor, and stop functions.")
                return run_de_nocompile(mutation_type,pp, bounds, reject_sample_max, max_iters, f, p_best, cr, seed, crf,stop_apply, pop_eval,monitor_func, *stack_mem, *eval_opts)
    else:
        print('Specified mutation strategy is not implemented for this optimizer. Please select an available strategy.')
        return None,None

@register_jitable(inline=True,**cmn.nb_pcs())
def init_algo_stack(population:A, yes=True):
    nt=nb.get_num_threads()
    m_pop=np.empty(population.shape,dtype=np.float64)
    _crr, _t_pop = init_parameter_stack(nt, population.shape[1], yes)
    _idx,_ftdiym=init_population_stack(nt, population.shape[0])
    return m_pop,_idx[:-1],_crr,_t_pop,_ftdiym,_idx[-1],#_adegen


@register_jitable(inline=True,**cmn.nb_pcs())
def init_parameter_stack(nt, pop_dim, yes=True):
    _crr = np.empty((nt,pop_dim),dtype=np.int64)
    _t_pop=np.empty((nt,pop_dim),dtype=np.float64) if yes else None
    return _crr,_t_pop


@register_jitable(inline=True,**cmn.nb_pcs())
def init_population_stack(nt, pop_size):
    _idx = np.empty((nt + 1, pop_size), dtype=np.int64)
    _idx[:-1] = np.arange(0, pop_size)
    _ftdiym = np.empty((pop_size,), dtype=np.float64)
    #_ftdiym[:]=0. #if there is dependence on prev fit values, then zero init is necessary
    return _idx,_ftdiym


def _ini_spc(init_spec,pop_dim,bounds):
    if type(init_spec)==np.ndarray:
        p=init_spec
        pop_dim[0],pop_dim[1]=p.shape[0],p.shape[1]
    else:
        p = cmn.init_randuniform(pop_dim[0], pop_dim[1], bounds[:pop_dim[1]])
    return p

@overload(_ini_spc,**cmn.nb_cs())
def _ini_spc_(init_spec,pop_dim,bounds):
    if isinstance(init_spec,types.Array):
        def _i(init_spec,pop_dim,bounds):
            p = init_spec
            pop_dim[0], pop_dim[1] = p.shape[0], p.shape[1]
            return p
    else:
        _i= lambda init_spec,pop_dim,bounds: cmn.init_randuniform(pop_dim[0], pop_dim[1], bounds[:pop_dim[1]])
    return _i


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
    #probably change so that the configs become a 2 element array with the same values later.
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

#---- DE SETTINGS OPTIMIZATION UTILS
#tldr; optimize DE settings for a specific class of problems using DE itself.
#While it's great to have a strong general optimization algorithm, it's possible to get much better performance on specific problems
#by providing optimal hyperparameters to the optimizer. For example let's say you have a large model with many parameters, and
#your dataset is also large such that a single evaluation now takes a long time. You might be able to select 1/100th or even 1/1000th
#of that dataset to teach the optimizer the model's parameter structure through it's tunable hyperparameters. This may increase the
#DE optimizers performance by an order of magnitude.
@register_jitable(**cmn.nb_pcs())
def default_settings_bounds(min_pop=4,max_pop=128):
    br=np.empty((7,2),dtype=np.float64)
    br[0]=0.01,3.
    br[1] = 0.01, 3.
    br[2]=0.0,1.0
    br[3] = 0.0, 1.0
    br[4] = 0.01, 1.0
    br[5] = 0.01, 1.0
    br[6] = min_pop, max_pop
    return br


@nb.njit(**cmn.nb_pcs()) #may barely be worth setting as njit
def vec_to_settings_cost(search_vec,bounds,f:np.ndarray,cr:np.ndarray,p_b:np.ndarray,pop_dim:np.ndarray):
    """Places all search scalars into the tunable settings of DE, handling the constraints.
    Also returns the L1 constraint violation cost."""
    sv=search_vec
    cstc=0.
    cs,f[0],f[1]=_hw_cost(bounds[0,0],bounds[0,1],bounds[1,0],bounds[1,1],sv[0],sv[1]) #f jitter range
    cstc+=cs
    cs,cr[0],cr[1]=_hw_cost(bounds[2,0],bounds[2,1],bounds[3,0],bounds[3,1],sv[2],sv[3]) #cr jitter range
    cstc+=cs
    cs,pb0,pb1=_hw_cost(bounds[4,0],bounds[4,1],bounds[5,0],bounds[5,1],sv[4],sv[5]) #p_best jitter range
    cstc += cs
    pd=min(max(bounds[6,0], sv[6]), bounds[6,1]) #pop_dim
    cstc += abs(sv[6]-pd)
    pmi=cmn.fastround_int(pd)
    p_b[0],p_b[1]=min(ceil(pb0*pd),pmi),min(ceil(pb1*pd),pmi)
    pop_dim[0]=pmi
    cstc/=(7.**.5) #technically there are 10 edges, but I think this makes sense.
    return cstc,pmi

@register_jitable(inline=True,**cmn.nb_pcs())
def _hw_cost(bn0,bx0,bn1,bx1,s0,s1): #hw -> halfway cost
    cstc=0.
    bp0 = min(max(bn0, s0), bx0)
    bp1 = min(max(bn1, s1), bx1)
    cstc += abs(s0 - bp0)/(bx0-bn0)
    cstc += abs(s1 - bp1)/(bx1-bn1)
    if s0 > s1:
        cstc += max(0., s0 - s1)*2./((bx1-bn1)+(bx0-bn0)) #in case the bounds have different scales.
        ms = (bp0 + bp1) / 2.
        return cstc, ms, ms
    else:
        return cstc, bp0, bp1

@register_jitable(**cmn.nb_pcs())
def get_tunable_settings(*all_opts): #REMEMBER to update if you add new things.
    al=all_opts
    """Returns (pop_arr, bounds, f, b_pest, cr)
    Positions within args: (1, 2, 5, 6, 7)
    Fitness constraint bounds aren't tunable, but could be randomized to some extent to increase robustness of tuning.
    Population isn't tunable directly but it's size is, so in case the user needs to see the size
    """
    #pop_arr unneeded as I'll store placeholder stack for max popsize, but including separately if other users want it.
    #If there are enough mutation types in the future you could add it
    #mutation_type,pp, bounds, reject_sample_max, max_iters, f, p_best, cr, seed, crf,stop_apply, pop_eval,monitor_func, *gus, *eval_opts
    return al[1],al[2],al[5],al[6],al[7]

@register_jitable(**cmn.nb_pcs())
def fix_pop_stack(n_pop,*all_opts):
    """all_opts sb the original with max pop_size. Receives new pop and opts and fixes DE stack."""
    al=all_opts
    #These settings will all be inited assuming max pop_size
    #idxs: pop 1, 14 _idx, 17 ftdiym, 18 _idxs
    nsz=n_pop.shape[0]
    ndx=al[14][:,:nsz]
    ftd =al[17][:nsz]
    ndxs = al[18][:nsz]
    return al[0],n_pop,*al[2:14],ndx,*al[15:17],ftd,ndxs,*al[19:]



#--- DE IMPLEMENTATIONS

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