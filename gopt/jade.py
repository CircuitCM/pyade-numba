import gopt.commons as cmn
import gopt.config as cfg
import numpy as np
from typing import Callable, Union, Dict, Any
from numba import types
import numba as nb
from numba.core.extending import overload,register_jitable
import math as m
import random as rand

from gopt.commons import place_randnormal

A=np.ndarray
NT=type(None)


def get_default_params(dim: int) -> dict:
    """
        Returns the default parameters of the JADE Differential Evolution Algorithm.
        :param dim: Size of the problem (or individual).
        :type dim: int
        :return: Dict with the default parameters of the JADE Differential
        Evolution Algorithm.
        :rtype dict
        """
    pop_size = 10 * dim
    return {'max_evals': 10000 * dim, 'individual_size': dim, 'callback': None,
            'population_size': pop_size, 'c': 0.1, 'p': max(.05, 3/pop_size), 'seed': None}


def apply(population_size: int, individual_size: int, bounds: np.ndarray,
          func: Callable[[np.ndarray], float], opts: Any,
          p: Union[int, float], c: Union[int, float], callback: Callable[[Dict], Any],
          max_evals: int, seed: Union[int, None]) -> [np.ndarray, int]:
    """
    Applies the JADE Differential Evolution algorithm.
    :param population_size: Size of the population.
    :type population_size: int
    :param individual_size: Number of gens/features of an individual.
    :type individual_size: int
    :param bounds: Numpy ndarray with individual_size rows and 2 columns.
    First column represents the minimum value for the row feature.
    Second column represent the maximum value for the row feature.
    :type bounds: np.ndarray
    :param func: Evaluation function. The function used must receive one
     parameter.This parameter will be a numpy array representing an individual.
    :type func: Callable[[np.ndarray], float]
    :param opts: Optional parameters for the fitness function.
    :type opts: Any type.
    :param p: Parameter to choose the best vectors. Must be in (0, 1].
    :type p: Union[int, float]
    :param c: Variable to control parameter adoption. Must be in [0, 1].
    :type c: Union[int, float]
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
    :param max_evals: Number of evaluations after the algorithm is stopped.
    :type max_evals: int
    :param seed: Random number generation seed. Fix a number to reproduce the
    same results in later experiments.
    :type seed: Union[int, None]
    :return: A pair with the best solution found and its fitness.
    :rtype [np.ndarray, int]
    """
    # 0. Check parameters are valid
    if type(population_size) is not int or population_size <= 0:
        raise ValueError("population_size must be a positive integer.")

    if type(individual_size) is not int or individual_size <= 0:
        raise ValueError("individual_size must be a positive integer.")

    if type(max_evals) is not int or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

    if type(bounds) is not np.ndarray or bounds.shape != (individual_size, 2):
        raise ValueError("bounds must be a NumPy ndarray.\n"
                         "The array must be of individual_size length. "
                         "Each row must have 2 elements.")

    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

    if type(p) not in [int, float] and 0 < p <= 1:
        raise ValueError("p must be a real number in (0, 1].")
    if type(c) not in [int, float] and 0 <= c <= 1:
        raise ValueError("c must be an real number in [0, 1].")

    np.random.seed(seed)

    #For later, a

    # 1. Init population
    population = gopt.commons.init_population(population_size, individual_size, bounds)
    u_cr = 0.5
    u_f = 0.6

    p = np.ones(population_size) * p
    fitness = gopt.commons.apply_fitness(population, func, opts)
    max_iters = max_evals // population_size
    for current_generation in range(max_iters):
        # 2.1 Generate parameter values for current generation
        cr = np.random.normal(u_cr, 0.1, population_size)
        f = np.random.rand(population_size // 3) * 1.2
        f = np.concatenate((f, np.random.normal(u_f, 0.1, population_size - (population_size // 3))))

        # 2.2 Common steps
        mutated = gopt.commons.current_to_pbest_mutation(population, fitness, f.reshape(len(f), 1), p, bounds)
        crossed = gopt.commons.crossover(population, mutated, cr.reshape(len(f), 1))
        c_fitness = gopt.commons.apply_fitness(crossed, func, opts)
        population, indexes = gopt.commons.selection(population, crossed,
                                                     fitness, c_fitness, return_indexes=True)

        # 2.3 Adapt for next generation
        if len(indexes) != 0:
            u_cr = (1 - c) * u_cr + c * np.mean(cr[indexes])
            u_f = (1 - c) * u_f + c * (np.sum(f[indexes]**2) / np.sum(f[indexes]))

        fitness[indexes] = c_fitness[indexes]
        if callback is not None:
            callback(**(locals()))

    best = np.argmin(fitness)
    return population[best], fitness[best]

def _ini_non_compiled_settings():pass

@register_jitable(**cfg.jit_s)
def init_algo_stack(population:A, yes=True):
    nt=nb.get_num_threads()
    m_pop=np.empty(population.shape,dtype=np.float64)
    _crr, _t_pop = init_parameter_stack(nt, population.shape[1], yes)
    _idx,_ftdiym,_f_sam,_cr_sam=init_population_stack(nt, population.shape[0])
    return m_pop,_idx[:-1],_crr,_t_pop,_ftdiym,_f_sam,_cr_sam #_idx[-1],_f_sam,_cr_sam


@register_jitable(**cfg.jit_s)
def init_parameter_stack(nt, pop_dim, yes=True):
    _crr = np.empty((nt,pop_dim),dtype=np.int64)
    _t_pop=np.empty((nt,pop_dim),dtype=np.float64) if yes else None
    return _crr,_t_pop


@register_jitable(**cfg.jit_s)
def init_population_stack(nt, pop_size):
    _idx = np.empty((nt + 1, pop_size), dtype=np.int64)
    _idx[:-1] = np.arange(0, pop_size)
    _ftdiym = np.empty((pop_size,), dtype=np.float64)
    _f_sam = np.empty((pop_size,), dtype=np.float64)
    _cr_sam = np.empty((pop_size,), dtype=np.float64)
    #_ftdiym[:]=0. #if there is dependence on prev fit values, then zero init is necessary
    return _idx,_ftdiym,_f_sam,_cr_sam

#implement when you want to do meta optimization.
def set_concurrentrun_stack():pass
def fix_concurrentrun_stack():pass
def fix_stack():pass

@nb.njit(**cfg.jit_s)
def run_jade(mutation_selector:int,population: A,
                            bounds: A,
                      reject_mx:int,
                            max_evals:int,
                            f_init: float,f_bds: A,f_scl:float, f_c:float, f_genc:NT|Any, #c adaption rate, scl is std/var.
                            cr_init: float, cr_bds: A, cr_scl:float, cr_c:float, cr_genc:NT|Any,
                            p:int|np.ndarray, #maybe extend to p group later.
                            leh_order:float, #lehman mean order default 2 for contraharmonic. biases higher, which is good..
                            cross_apply: Callable,
                            stop_apply: Callable,
                            pop_eval: Callable,
                            monitor:Callable,
                            _m_pop:A,_idx:A,_crr:A,_t_pop:A,_ftdiym:A, _f_sam:A, _cr_sam:A,
                            *eval_opts) -> tuple[A,A]:
    return run_jade_nocompile(mutation_selector,population,bounds,reject_mx,max_evals,f_init,f_bds,f_scl, f_c,f_genc,cr_init, cr_bds, cr_scl, cr_c,cr_genc,p,leh_order,cross_apply,stop_apply,pop_eval,monitor,_m_pop,_idx,_crr,_t_pop,_ftdiym, _idxs,*eval_opts)


@register_jitable(**cfg.jit_s)
def run_jade_nocompile(mutation_selector:int,population: A,
                            bounds: A,
                      reject_mx:int,
                            max_evals:int,
                            f_init: float,f_bds: A,f_scl:float, f_c:float, f_genc:NT|Any,
                            cr_init: float, cr_bds: A, cr_scl:float, cr_c:float, cr_genc:NT|Any,
                            p:int|np.ndarray, #maybe extend to p group later.
                            leh_order:float, #lehman mean order default 2 for contraharmonic. biases higher, which is good..
                            cross_apply: Callable,
                            stop_apply: Callable,
                            pop_eval: Callable,
                            monitor:Callable,
                            _m_pop:A,_idx:A,_crr:A,_t_pop:A,_ftdiym:A,  _f_sam:A, _cr_sam:A,
                            *eval_opts) -> tuple[A,A]:
    #maybe pbest and pbesta2 later.
    max_iters = m.ceil(max_evals / population.shape[0])
    _jade_c_t_pbest(population,bounds,reject_mx,max_iters,f_init,f_bds,f_scl, f_c,f_genc,cr_init, cr_bds, cr_scl, cr_c,cr_genc,p,leh_order,cross_apply,stop_apply,pop_eval,monitor,_m_pop,_idx,_crr,_t_pop,_ftdiym, _idxs,*eval_opts)
    _b = np.argmin(_ftdiym)
    return population[_b], _ftdiym[_b]

#For mutation strategies you'll only use the pbests so shouldn't need separate functions tbh.
@register_jitable(**cfg.jit_s)
def _jade_c_t_pbest(population: A,
                            bounds: A,
                      reject_mx:int,
                            max_iters:int,
                            f_init: float,f_bds: A,f_scl:float, f_c:float, f_genc:NT|float,
                            cr_init: float, cr_bds: A, cr_scl:float, cr_c:float, cr_genc:NT|float,
                            p:int|np.ndarray, #maybe extend to p adaption later.
                            leh_order:float, #lehmer mean order default 2 for contraharmonic. biases higher, which is good..
                            cross_apply: Callable,
                            stop_apply: Callable,
                            pop_eval: Callable,
                            monitor:Callable,
                            _m_pop:A,_idx:A,_crr:A,_t_pop:A,_ftdiym:A,  _f_sam:A, _cr_sam:A,
                            *eval_opts):

    #Think about making c decay quicker and randomization std larger based on frequency of new bests relative to total evals.
    #But that would be your own separate algo.
    #By including p adaptation in you may effectively be incorperating a similar feature to what sade does.
    #makes it a current to rand mixture.
    u_f,u_cr,tp,idx=_first(population,_ftdiym,f_init,_f_sam,f_bds,f_scl,cr_init,_cr_sam,cr_bds,cr_scl,p,pop_eval,*eval_opts)
    for current_generation in range(max_iters):
        cmn.uchoice_mutator(population, _m_pop, _cr_sam, bounds, reject_mx, cross_apply,
                            cmn.c_to_pbest_mutate, cfg._C_T_PB_M_R,_idx, _crr, _t_pop, idx, tp, _f_sam)
        u_f,f_genc,u_cr,cr_genc,tp,idx=_the_rest(population,_m_pop,_ftdiym,idx,u_f,_f_sam,f_bds,f_scl,f_c,f_genc,u_cr,_cr_sam,cr_bds,cr_scl,cr_c,cr_genc,p,leh_order,pop_eval,monitor,*eval_opts)
        if stop_apply is not None and stop_apply(population,_ftdiym):break

@register_jitable(**cfg.jit_s)
def _first(population,_ftdiym,f,f_sam,f_bds,f_scl,cr,cr_sam,cr_bds,cr_scl,p,pop_eval,*eval_opts):
    _ftdiym[:] = pop_eval(population, *eval_opts)
    idx = np.argsort(_ftdiym)
    p=_mk_paramsamples(population,f,f_sam,f_bds,f_scl,cr,cr_sam,cr_bds,cr_scl,p)
    return f,cr,p,idx

@register_jitable(**cfg.jit_s)
def _the_rest(population,m_pop,_ftdiym,idx,u_f,f_sam,f_bds,f_scl,f_c,f_genc,u_cr,cr_sam,cr_bds,cr_scl,cr_c,cr_genc,p,leh_order,pop_eval,monitor,*eval_opts):
    ftdiym = pop_eval(m_pop, *eval_opts)
    bdx = cmn.select_better(_ftdiym, ftdiym, idx)
    if bdx.shape[0] != 0:
        u_f, f_genc = optgen_ema(f_genc, f_c, u_f, (sum(f_sam[bdx] ** leh_order) / sum(f_sam[bdx] ** (leh_order - 1))))
        u_cr,cr_genc=optgen_ema(cr_genc,cr_c,u_cr,sum(cr_sam[bdx]) / bdx.shape[0])
    else:
        f_genc=disc_orn(f_genc,f_c)
        cr_genc =disc_orn(cr_genc, cr_c)
    cmn._meval(monitor, population, m_pop, _ftdiym, *eval_opts)
    _ftdiym[bdx] = ftdiym[bdx]
    population[bdx] = m_pop[bdx]
    idx = np.argsort(_ftdiym)  # always needs to be done, idx reusage, if improvements are very rare, might be a little more efficient to have a separate idx array
    p=_mk_paramsamples(population,u_f,f_sam,f_bds,f_scl,u_cr,cr_sam,cr_bds,cr_scl,p)
    return u_f,f_genc,u_cr,cr_genc,p,idx


@register_jitable(**cfg.jit_s)
def _mk_paramsamples(population,u_f,f_sam,f_bds,f_scl,u_cr,cr_sam,cr_bds,cr_scl,p):
    tp = cmn._pset(p)
    #The original implementation uses a 2/3rd normal 1/3rd uniform mixture, but the more modern version uses cauchy.
    #I implement it with a limited rejection cauchy bound, hard coded to 3 attempts, seems to be a decent distribution balance
    #between allowing the edges and resampling to not overbias.
    #with this version, min f ~.05 or > 0, is probably suitable.
    for i in range(population.shape[0]):  # cauchy and normal
        # original implementation is, cauchy centered at mean, .1 scale, if <0 regenerate if>1 set it to maximum.
        cr_sam[i] = min(cr_bds[0], max(rand.normalvariate(u_cr, cr_scl), cr_bds[1]))
        for n in range(3):
            f_sam[i] = u_f + m.tan(m.pi * (rand.random() - 0.5)) * f_scl  # min(fmn + m.tan(m.pi * (rand.random() - 0.5)) *
            if  f_bds[1] > f_sam[i] and f_sam[i] >= f_bds[0]:
                break
        f_sam[i] = max(min(f_sam[i], f_bds[1]), f_bds[0])
        #when population count is high, u_f might get more biased upwards, thanks to cauchy and lehmer, but maybe that is valid.
        #When there are greater frequencies of successes per generation u_f will also be biased higher, that makes sense.
    return tp


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






