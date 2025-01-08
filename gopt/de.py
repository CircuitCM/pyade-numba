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


def apply_de(pop_eval: Callable,
             f: float=None, #If None jitters between .5 and 1.
             cr: float=.9, #Original CR value for differential evolution.
             init_spec: Callable | str | np.ndarray = 'sobol',  #If 'sobol', or 'rand' selects from uniform. Otherwise provide array or initializer.
             pop_dim: tuple|list|np.ndarray|None=None, #If None, then init_spec needs to be a callable or array of (pop_sz, individual_sz).
             bounds: A = None, #Required if you don't init your own population/supply a callable that uses pop and indiv size.
             max_evals: int = 10000,
             cross_type:int=CrossoverSelector.BIN,#for now its only bin. can add others if helpful in the future.
             mutation_type:int=MutationSelector.CUR_T_PBEST,
             enf_bounds: bool = False, #Recommended to keep it False, and truncate parameters to boundaries in your pop_eval.
             reject_sample_max:int= None, #Must be >1 or None, If None then no rejection_sampling. Can reduce boundary bias, and increase meaningful search in valid boundaries for same # population evals.
             p_best:float=.11, #Only used if a probability wgt'd mutator is used. jitters between (0,.5] if None.
             seed:int=42_420_69_9001,
             stop_condition:Callable|float|None=None,
             eval_opts:Sequence=()) -> tuple[A,A]:
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

    if (type(f) is not int and type(f) is not float) or not 0 <= f <= 2:
        raise ValueError("f (mutation parameter) must be a "
                         "real number in [0,2].")

    if (type(cr) is not int and type(cr) is not float) or not 0 <= cr <= 1:
        raise ValueError("cr (crossover ratio) must be a "
                         "real number in [0,1].")

    if type(max_evals) is not int or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

    # if type(bounds) is not A or bounds.shape != (individual_size, 2):
    #     raise ValueError("bounds must be a NumPy ndarray.\n"
    #                      "The array must be of individual_size length. "
    #                      "Each row must have 2 elements.")

    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

    if pop_dim is not None:
        pp = cmn.init_population(pop_dim[0],pop_dim[1], bounds,init_spec)
    else:
        pp = cmn.init_population(None,None, bounds, init_spec)

    if isinstance(p_best,float): #Then also assume it's between 0 and 1 no user check... maybe should add if it causes memory leaks tho.
        p_best=ceil(pp.shape[0]*p_best)

    enf_bounds=enf_bounds and bounds
    gus=_ini_h(pp,False if not bounds or reject_sample_max is None else True)
    max_iters=ceil(max_evals/pp.shape[0]) #a little more than max.
    crf=cmn.bin_mutate if cross_type == CrossoverSelector.BIN else None #implement later if needed
    stop_apply= cmn.mk_fitness_thresh_stop(stop_condition) if type(stop_condition) is float else stop_condition #leaving none or callable.
    match (mutation_type,cross_type):
        case MutationSelector.BINARY,_:
            return _de_bin_mutate_bc(pp,bounds,enf_bounds,reject_sample_max,max_iters,f,cr,seed,crf,stop_apply,pop_eval,*gus,*eval_opts)
        case MutationSelector.CUR_T_BEST2, _:
            return _de_c_t_best2_bc(pp, bounds, enf_bounds, reject_sample_max, max_iters, f, cr, seed, crf,stop_apply, pop_eval, *gus, *eval_opts)
        case MutationSelector.CUR_T_PBEST, _:
            return _de_c_t_pbest_bc(pp, bounds, enf_bounds, reject_sample_max, max_iters, f, p_best, cr, seed, crf,stop_apply, pop_eval, *gus, *eval_opts)


def _ini_h(population:A,yes=True):
    nt=nb.get_num_threads()
    m_pop=np.empty(population.shape,dtype=np.float64)
    _crr = np.empty((nt,population.shape[1]),dtype=np.int64)
    _t_pop=np.empty((nt,population.shape[1]),dtype=np.float64) if yes else None
    _idx = np.empty((nt+1, population.shape[0]), dtype=np.int64)
    _idx[:-1]= np.arange(0,population.shape[0])
    _ftdiym=np.empty((population.shape[0],), dtype=np.float64)
    return m_pop,_idx[:-1],_crr,_t_pop,_ftdiym,_idx[-1]


@cmn.compile_with_fallback
def _de_bin_mutate_bc(population: A,
                            bounds: A, enf_bounds:bool, #Enforce bounds
                      reject_mx:int,
                            max_iters:int,
                            f: float, cr: float, #if f==-1. then jitter along .5 1.
                            seed: int,
                            cross_apply: Callable,
                            stop_apply:Callable,
                            pop_eval: Callable,
                            _m_pop:A,_idx:A,_crr:A,_t_pop:A,_ftdiym:A, _idxs:A,
                            *eval_opts) -> tuple[A,A]:
    tf=_first(population,_ftdiym,seed,f,pop_eval,*eval_opts)
    for current_generation in range(max_iters):
        cmn.uchoice_mutator(population, _m_pop,cr, bounds, enf_bounds, reject_mx, cross_apply, cmn.bin_mutate, cfg.BIN_MUTATE_R, _idx, _crr, _t_pop, tf)
        tf=_the_rest(population,_m_pop,_ftdiym,_idxs,f,pop_eval,*eval_opts)
        if stop_apply is not None and stop_apply(population,_ftdiym):break
    #Old behavior, returns best from last gen, the user should actually track progress within the pop_eval callable.
    _b = np.argmin(_ftdiym)
    return population[_b], _ftdiym[_b]

@cmn.compile_with_fallback
def _de_c_t_best2_bc(population: A,
                            bounds: A, enf_bounds:bool, #Enforce bounds
                      reject_mx:int,
                            max_iters:int,
                            f: float, cr: float, #if f==-1. then jitter along .5 1.
                            seed: int,
                            cross_apply: Callable,
                            stop_apply: Callable,
                            pop_eval: Callable,
                            _m_pop:A,_idx:A,_crr:A,_t_pop:A,_ftdiym:A, _idxs:A,
                            *eval_opts) -> tuple[A,A]:
    tf=_first(population,_ftdiym,seed,f,pop_eval,*eval_opts)
    _b = np.argmin(_ftdiym)
    for current_generation in range(max_iters):
        cmn.uchoice_mutator(population, _m_pop,cr, bounds, enf_bounds, reject_mx, cross_apply, cmn.c_to_best_2_bin_mutate, cfg.C_T_B2_MUTATE_R, _idx, _crr, _t_pop, _b,tf)
        tf=_the_rest(population,_m_pop,_ftdiym,_idxs,f,pop_eval,*eval_opts)
        _b = np.argmin(_ftdiym)
        if stop_apply is not None and stop_apply(population,_ftdiym):break
    return population[_b], _ftdiym[_b]

@cmn.compile_with_fallback
def _de_c_t_pbest_bc(population: A,
                            bounds: A, enf_bounds:bool, #Enforce bounds
                      reject_mx:int,
                            max_iters:int,
                            f: float, p:int, cr: float, #if f==-1. then jitter along .5 1.
                            seed: int,
                            cross_apply: Callable,
                            stop_apply: Callable,
                            pop_eval: Callable,
                            _m_pop:A,_idx:A,_crr:A,_t_pop:A,_ftdiym:A, _idxs:A,
                            *eval_opts) -> tuple[A,A]:
    tf=_first(population,_ftdiym,seed,f,pop_eval,*eval_opts)
    _b = np.argsort(_ftdiym) #this will spam new arrays, boohoo
    psz= population.shape[0] // 2
    for current_generation in range(max_iters):
        tp=_pset(p,psz)
        cmn.uchoice_mutator(population, _m_pop,cr, bounds, enf_bounds, reject_mx, cross_apply, cmn.c_to_pbest_mutate, cfg.C_T_PB_MUTATE_R, _idx, _crr, _t_pop, _b, tp, tf)
        tf=_the_rest(population,_m_pop,_ftdiym,_idxs,f,pop_eval,*eval_opts)
        _b = np.argsort(_ftdiym)
        if stop_apply is not None and stop_apply(population,_ftdiym):break
    _b = np.argmin(_ftdiym)
    return population[_b], _ftdiym[_b]


@nb.njit(**cmn.nb_pcs())
#@nb.njit(inline='always')
def _first(population,_ftdiym,seed,f,pop_eval,*eval_opts):
    rand.seed(seed) #Assuming it can handle None correctly?
    _ftdiym[:] = pop_eval(population, *eval_opts)  # fitness array sb embedded into eval_opts already. non _'d
    return _fset(f)

@nb.njit(**cmn.nb_pcs())
#@nb.njit(inline='always')
def _the_rest(population,m_pop,_ftdiym,_idxs,f,pop_eval,*eval_opts):
    ftdiym = pop_eval(m_pop, *eval_opts)
    bdx=cmn.select_better(_ftdiym,ftdiym,_idxs)
    _ftdiym[bdx]=ftdiym[bdx]
    population[bdx]=m_pop[bdx]
    return _fset(f)

def _fset(f):pass
@overload(_fset,inline='always')
def _fset_(f):
    if isinstance(f,types.NoneType):
        def _f(f): return rand.uniform(.5, 1.) #jitter f
        return _f
    else:
        def _f(f): return f
        return _f

def _pset(p,pr):pass
@overload(_pset,inline='always')
def _pset_(p,pr):
    if isinstance(p,types.NoneType):
        def _f(p,pr): return rand.randint(1, pr) #jitter p for pbest selection
        return _f
    else:
        def _f(p,pr): return p
        return _f
