import gopt.commons as cmn
import gopt.config as cfg
import numpy as np
from typing import Callable, Union, Dict, Any
import numba as nb
A=np.ndarray


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

#For mutation strategies you'll only use the pbests so shouldn't need separate functions tbh.
@nb.njit(**cmn.nb_cs())
def _jade_c_t_pbest_bc(population: A,
                            bounds: A, enf_bounds:bool, #Enforce bounds
                      reject_mx:int,
                            max_iters:int,
                            f_init: float, p:int|np.ndarray, cr_init: float, #if f==-1. then jitter along .5 1.
                            f_bds: np.ndarray,
                            cr_bds: np.ndarray,
                            c:float, #adaption rate
                            leh_order:float, #lehman mean order default 2 for contraharmonic. biases higher, which is good..
                            seed: int,
                            cross_apply: Callable,
                            stop_apply: Callable,
                            pop_eval: Callable,
                            monitor:Callable,
                            _m_pop:A,_idx:A,_crr:A,_t_pop:A,_ftdiym:A, _idxs:A,
                            *eval_opts) -> tuple[A,A]:

    #Think about making c decay quicker and randomization std larger based on frequency of new bests relative to total evals.
    #But that would be your own separate algo.
    u_cr = cr_init
    u_f = f_init
    cmn._rset(seed)
    _ftdiym[:] = pop_eval(population, *eval_opts)  # fitness array sb embedded into eval_opts already. non _'d
    _b = np.argsort(_ftdiym)
    for current_generation in range(max_iters):
        tp=cmn._pset(p)
        #tp=pset(p,psz)
        cmn.uchoice_mutator(population, _m_pop, cr, bounds, enf_bounds, reject_mx, cross_apply,
                            cmn.c_to_pbest_mutate, cfg._C_T_PB_M_R,
                            _idx, _crr, _t_pop, _b, tp, tf)
        ftdiym = pop_eval(_m_pop, *eval_opts)
        # To capture the most information, monitor comes right before new pop transition.
        monitor(population, _m_pop, _ftdiym, *eval_opts)
        bdx = cmn.select_better(_ftdiym, ftdiym, _idxs)
        if len(bdx) != 0:
            u_cr = (1 - c) * u_cr + c * np.mean(cr[indexes])
            u_f = (1 - c) * u_f + c * (np.sum(f[indexes]**2) / np.sum(f[indexes]))
        _ftdiym[bdx] = ftdiym[bdx]
        population[bdx] = _m_pop[bdx]
        tf=cmn._fset(f)
        if stop_apply is not None and stop_apply(population,_ftdiym):break
        _b = np.argsort(_ftdiym)
    _b = np.argmin(_ftdiym)
    return population[_b], _ftdiym[_b]
