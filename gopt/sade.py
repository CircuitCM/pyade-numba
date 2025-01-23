import gopt.commons as cmn
import gopt.config as cfg
import numpy as np
from typing import Callable, Union, Dict, Any
import numba as nb
from numba.core.extending import overload,register_jitable
import math as m
import random as rand
A=np.ndarray


def get_default_params(dim: int) -> dict:
    """
    Returns the default parameters of the Self-adaptive Differential Evolution Algorithm (SaDE).
    :param dim: Size of the problem (or individual).
    :type dim: int
    :return: Dict with the default parameters of SaDe
    :rtype dict
    """
    return {'max_evals': 10000 * dim, 'population_size': 10 * dim, 'callback': None,
            'individual_size': dim, 'seed': None, 'opts': None}


def apply(population_size: int, individual_size: int,
          bounds: np.ndarray,
          func: Callable[[np.ndarray], float], opts: Any,
          callback: Callable[[Dict], Any],
          max_evals: int, seed: Union[int, None]):
    """
    Applies the Self-adaptive differential evolution algorithm (SaDE).
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
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
    :param max_evals: Number of evaluatios after the algorithm is stopped.
    :type max_evals: int
    :param seed: Random number generation seed. Fix a number to reproduce the
    same results in later experiments.
    :type seed: Union[int, None]
    :return: A pair with the best solution found and its fitness.
    :rtype [np.ndarray, int]
    """

    if type(population_size) is not int or population_size <= 0:
        raise ValueError("population_size must be a positive integer.")

    if type(individual_size) is not int or individual_size <= 0:
        raise ValueError("individual_size must be a positive integer.")

    if type(bounds) is not np.ndarray or bounds.shape != (individual_size, 2):
        raise ValueError("bounds must be a NumPy ndarray.\n"
                         "The array must be of individual_size length. "
                         "Each row must have 2 elements.")

    if type(max_evals) is not int or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

    # 1. Initialization
    np.random.seed(seed)
    population = gopt.commons.init_population(population_size, individual_size, bounds)

    # 2. SaDE Algorithm
    probability = 0.5
    fitness = gopt.commons.apply_fitness(population, func, opts)
    cr_m = 0.5
    f_m = 0.5

    sum_ns1 = 0
    sum_nf1 = 0
    sum_ns2 = 0
    sum_nf2 = 0
    cr_list = []

    f = np.random.normal(f_m, 0.3, population_size)
    f = np.clip(f, 0, 2)

    cr = np.random.normal(cr_m, 0.1, population_size)
    cr = np.clip(cr, 0, 1)

    max_iters = max_evals // population_size
    for current_generation in range(max_iters):
        # 2.1 Mutation
        # 2.1.1 Randomly choose which individuals do each mutation
        choice = np.random.rand(population_size)
        choice_1 = choice < probability
        choice_2 = choice >= probability

        # 2.1.2 Apply the mutations
        mutated = population.copy()
        mutated[choice_1] = gopt.commons.binary_mutation(population[choice_1],
                                                         f[choice_1].reshape(sum(choice_1), 1), bounds)
        mutated[choice_2] = gopt.commons.current_to_best_2_binary_mutation(population[choice_2],
                                                                           fitness[choice_2],
                                                                           f[choice_2].reshape(sum(choice_2), 1),
                                                                           bounds)

        # 2.2 Crossover
        crossed = gopt.commons.crossover(population, mutated, cr.reshape(population_size, 1))
        c_fitness = gopt.commons.apply_fitness(crossed, func, opts)

        # 2.3 Selection
        population = gopt.commons.selection(population, crossed, fitness, c_fitness)
        winners = c_fitness < fitness
        fitness[winners] = c_fitness[winners]

        # 2.4 Self Adaption
        chosen_1 = np.sum(np.bitwise_and(choice_1, winners))
        chosen_2 = np.sum(np.bitwise_and(choice_2, winners))
        sum_ns1 += chosen_1
        sum_ns2 += chosen_2
        sum_nf1 += np.sum(choice_1) - chosen_1
        sum_nf2 += np.sum(choice_2) - chosen_2
        cr_list = np.concatenate((cr_list, cr[winners]))

        # 2.4.1 Adapt mutation strategy probability
        if (current_generation + 1) % 50 == 0:
            probability = sum_ns1 * (sum_ns2 + sum_nf2) / (sum_ns2 * (sum_ns1 + sum_nf1))
            probability = np.clip(probability, 0, 1)
            if np.isnan(probability):
                probability = .99
            sum_ns1 = 0
            sum_ns2 = 0
            sum_nf1 = 0
            sum_nf2 = 0

        # 2.4.2
        if (current_generation + 1) % 25 == 0:
            if len(cr_list) != 0:
                cr_m = np.mean(cr_list)
                cr_list = []
            cr = np.random.normal(cr_m, 0.1, population_size)
            cr = np.clip(cr, 0, 1)

        if callback is not None:
            callback(**(locals()))

    best = np.argmin(fitness)
    return population[best], fitness[best]

@register_jitable(**cfg.jit_s)
def init_algo_stack(population:A, yes=True):
    nt=nb.get_num_threads()
    m_pop=np.empty(population.shape,dtype=np.float64)
    _crr, _t_pop = init_parameter_stack(nt, population.shape[1], yes)
    _idx,_ftdiym=init_population_stack(nt, population.shape[0])
    return m_pop,_idx[:-1],_crr,_t_pop,_ftdiym,_idx[-1],#_adegen


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
    #_ftdiym[:]=0. #if there is dependence on prev fit values, then zero init is necessary
    return _idx,_ftdiym

#implement when you want to do meta optimization.
def set_concurrentrun_stack():pass
def fix_concurrentrun_stack():pass
def fix_stack():pass

@nb.njit(**cfg.jit_s)
def run_sade(population: A,
                       bounds: A,
                       reject_mx: int,
                       max_iters: int,
                       f: float, f_std: float,
                       cr: float, cr_std: float, cr_freq: float,  # add bounds if you actually want to develop this algo more
                       t_prob: int | np.ndarray, p_freq: int,
                       cross_apply: Callable,
                       stop_apply: Callable,
                       pop_eval: Callable,
                       monitor: Callable,
                       _m_pop: A, _idx: A, _crr: A, _t_pop: A, _ftdiym: A, _idxs: A,
                       *eval_opts):

    return run_sade_nocompile(population,bounds,reject_mx,max_iters,f,f_std,cr,cr_std, cr_freq,t_prob,p_freq,cross_apply,stop_apply,pop_eval,monitor,
                                     _m_pop,_idx,_crr,_t_pop,_ftdiym, _idxs,*eval_opts)



@register_jitable(**cfg.jit_s)
def run_sade_nocompile(population: A,
                       bounds: A,
                       reject_mx: int,
                       max_iters: int,
                       f: float, f_std: float,
                       cr: float, cr_std: float, cr_freq: float,  # add bounds if you actually want to develop this algo more
                       t_prob: int | np.ndarray, p_freq: int,
                       cross_apply: Callable,
                       stop_apply: Callable,
                       pop_eval: Callable,
                       monitor: Callable,
                       _m_pop: A, _idx: A, _crr: A, _t_pop: A, _ftdiym: A, _idxs: A,
                       *eval_opts):
    # The original sade, though I like the emas of jade instead of the generation sampling method for cr. Yeah this CR finding method seems silly.
    # Using the same randomly selected CR values for several generations, instead of generating them from the same distribution at each iteration
    # Could leave gaps in convergence runs.
    # Also rand1/binary could end up with some more duplicates
    # sade uses current-to-best-2 but in the paper it appears no different from current to best...
    # https://www.sciencedirect.com/science/article/pii/S2351978920308684 according to this sade isn't competitive for tensile optimization.
    cr_sam = np.minimum(np.maximum(0., np.random.normal(cr, cr_std, population.shape[0])),
                        1.)  # tbh could choose more than pop size of cr and random select.
    f_sam = np.random.normal(f, f_std, population.shape[0])  # maybe turned these into arg arrays for further development too.
    s_bin = s_tbi = 0
    s_best = s_tbe = 0
    cr_sbest = 0
    cr_soccur = 0
    # u_f = f_mean #f doesn't adapt in this one.

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1554904 implementation
    ftdiym = pop_eval(population, *eval_opts)
    _ftdiym[:] = ftdiym
    _b = np.argmin(_ftdiym)  # rand1
    for current_generation in range(max_iters):
        bn = cmn.fast_binom(population.shape[0], t_prob)
        cmn.durstenfeld_p_shuffle(cr_sam, cr_sam.shape[0])
        s_bin += bn  # total sums, not sums are redundant.
        s_best += (population.shape[0] - bn)
        cmn.uchoice_mutator(population[:bn], _m_pop[:bn], cr_sam[:bn], bounds, reject_mx, cross_apply,
                            cmn.c_to_pbest_mutate, cfg._BIN_M_R, _idx, _crr, _t_pop, f_sam)
        cmn.uchoice_mutator(population[bn:], _m_pop[bn:], cr_sam[bn:], bounds, reject_mx, cross_apply,
                            cmn.c_to_best_2_mutate, cfg._C_T_B2_M_R, _idx, _crr, _t_pop, _b, f_sam)

        ftdiym = pop_eval(_m_pop, *eval_opts)
        cmn._meval(monitor, population, _m_pop, _ftdiym, *eval_opts)
        bdx = cmn.select_better(_ftdiym, ftdiym, _idxs)
        _ftdiym[bdx] = ftdiym[bdx]
        population[bdx] = _m_pop[bdx]

        k1 = sum(bdx < bn)
        s_tbi += k1  # successes count for rand1
        s_tbe += (bdx.shape[0] - k1)  # success count for best 1 (or 2 paper doesn't say the difference)
        cr_sbest += sum(cr_sam[bdx])
        cr_soccur += bdx.shape[0]

        if (current_generation + 1) % p_freq == 0 and s_tbi > 0 and s_tbe > 0:
            t_prob = s_tbi * s_best / (s_tbi * s_best + s_tbe * s_bin)
            s_tbi = s_tbe = s_best = s_bin = 0

        if (current_generation + 1) % cr_freq == 0 and cr_soccur > 0:
            cr = cr_sbest / cr_soccur
            cr_sbest = cr_soccur = 0
            cmn.place_randnormal_fbds(cr_sam, cr, cr_std, 0., 1.)

        if stop_apply is not None and stop_apply(population, _ftdiym): break
        cmn.place_randnormal(f_sam, f, f_std, 0., 1.)
        _b = np.argmin(_ftdiym)
