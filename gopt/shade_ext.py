import gopt.commons as cmn
import gopt.config as cfg
import gopt.calculations as calc
import numpy as np
from typing import Callable, Union, Dict, Any
from numba import types
import numba as nb
from numba.core.extending import overload,register_jitable
import math as m
import random as rand

A=np.ndarray
NT=type(None)

#https://ieeexplore.ieee.org/document/6557798
#"An improved version of JADE."
#LSHADE and ILSHADE seem to be pretty simple modifications to SHADE that don't require changes to memory or the algo's structure.
#LSHADEepsin,cnepsin, salshade, different story.

class SHADEstrategy:
    SHADE=0
    LSHADE=1
    ILSHADE=2

class SHADEbase:

    def __init__(self, pop_min=4, scale_power=1., f_init=.5,f_mem=100, cr_init=.5, cr_mem=100, p_best=None):
        #default population start size = num dims * 18
        self.f_init=f_init
        self.f_mem=f_mem
        self.cr_init=cr_init
        self.cr_mem=cr_mem
        self.pop_min = pop_min
        self.scale_power = scale_power
        self.p_best=np.array([.05,.2]) if p_best is None else p_best


class SHADEsettings(SHADEbase):

    def __init__(self,f_init=.5,f_mem=100, cr_init=.5, cr_mem=100,p_best=None):
        #original shade set H or mem_size to 100.
        super().__init__(None,None,f_init,f_mem,cr_init,cr_mem,p_best)

class LSHADEsettings(SHADEbase):

    def __init__(self,pop_min=4,scale_power=1.,f_init=.5,f_mem=100, cr_init=.5, cr_mem=100,p_best=None):
        # according to ilshade paper, lshade was set to mem=6...
        super().__init__(pop_min,scale_power,f_init,f_mem,cr_init,cr_mem,p_best)

class ILSHADEsettings(SHADEbase):

    def __init__(self,pop_min=4,scale_power=1.,f_init=.8,f_mem=100,cr_init=.8,cr_mem=100,p_bestmin=.1,p_bestmax=.2):
        #maybe mem=6, idk for sure though.
        super().__init__(pop_min,scale_power,f_init,f_mem,cr_init,cr_mem,p_best=np.array([p_bestmin,p_bestmax]))



def get_default_params(dim: int):
    """
        Returns the default parameters of the SHADE Differential Evolution Algorithm.
        :param dim: Size of the problem (or individual).
        :type dim: int
        :return: Dict with the default parameters of the SHADE Differential
        Evolution Algorithm.
        :rtype dict
    """
    return {'max_evals': 10000 * dim, 'memory_size': 100,
            'individual_size': dim, 'population_size': 10 * dim,
            'callback': None, 'seed': None, 'opts': None}


def apply(population_size: int, individual_size: int, bounds: np.ndarray,
          func: Callable[[np.ndarray], float], opts: Any,
          memory_size: int, callback: Callable[[Dict], Any],
          max_evals: int, seed: Union[int, None]) -> [np.ndarray, int]:
    """
    Applies the SHADE differential evolution algorithm.
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
    :param memory_size: Size of the internal memory.
    :type memory_size: int
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

    #implementing only with archive, that seems to be established as the better pbest system overall.
    #SHADE is a bit like giving JADE the ability to self adapt it's std through variance in it's memory array
    #change from ema to memory length, and given memory is randomly sampled, don't need fancy ring indexing, just ct then modulo to 0.
    #successes for cr and f now have a L1 fitness improvement weighting scheme.
    #Memory defaults are 100 but might be worth changing that as well, eg to like 5-20 range, given JADE ema success ranges.
    #Also might be worth turning scale down from .1 to like .08, given the extra diversity the memory array will offer.

    np.random.seed(seed)
    random.seed(seed)

    # 1. Initialization
    population = gopt.commons.init_population(population_size, individual_size, bounds)
    m_cr = np.ones(memory_size) * 0.5
    m_f = np.ones(memory_size) * 0.5
    archive = []
    k = 0
    fitness = gopt.commons.apply_fitness(population, func, opts)

    all_indexes = list(range(memory_size))
    max_iters = max_evals // population_size
    for current_generation in range(max_iters):
        #Fortunately it seems to be mostly jade
        # 2.1 Adaptation
        r = np.random.choice(all_indexes, population_size)
        cr = np.random.normal(m_cr[r], 0.1, population_size)
        cr = np.clip(cr, 0, 1)
        cr[cr == 1] = 0
        f = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=population_size)
        f[f > 1] = 0 #wtf lol

        while sum(f <= 0) != 0: #well at least he tried to make it more efficient.
            r = np.random.choice(all_indexes, sum(f <= 0))
            f[f <= 0] = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=sum(f <= 0))

        p = np.random.uniform(low=2/population_size, high=0.2, size=population_size)

        # 2.2 Common steps
        mutated = gopt.commons.current_to_pbest_mutation(population, fitness, f.reshape(len(f), 1), p, bounds)
        crossed = gopt.commons.crossover(population, mutated, cr.reshape(len(f), 1))
        c_fitness = gopt.commons.apply_fitness(crossed, func, opts)
        population, indexes = gopt.commons.selection(population, crossed,
                                                     fitness, c_fitness, return_indexes=True)

        # 2.3 Adapt for next generation
        archive.extend(population[indexes])

        if len(indexes) > 0:
            if len(archive) > memory_size:
                archive = random.sample(archive, memory_size)
            if max(cr) != 0:
                weights = np.abs(fitness[indexes] - c_fitness[indexes])
                weights /= np.sum(weights)
                m_cr[k] = np.sum(weights * cr[indexes])
            else:
                m_cr[k] = 1

            m_f[k] = np.sum(f[indexes]**2)/np.sum(f[indexes])

            k += 1
            if k == memory_size:
                k = 0

        fitness[indexes] = c_fitness[indexes]
        if callback is not None:
            callback(**(locals()))

    best = np.argmin(fitness)
    return population[best], fitness[best]


@register_jitable(**cfg.jit_s)
def _shade_c_t_pbesta(population: A,
                      bounds: A,
                      reject_mx:int,
                      max_iters:int,
                      f_mem:A, f_bds: A, f_scl:float, f_leh_order:float,
                      cr_mem:A, cr_bds: A, cr_scl:float,  #lets assume no f_init as f_mem should already be initted.
                      p:int|np.ndarray,  #extend p to mem adaption.
                      cross_apply: Callable,
                      stop_apply: Callable,
                      pop_eval: Callable,
                      monitor:Callable,
                      _m_pop:A, _idx:A, _crr:A, _t_pop:A, _ftdiym:A, _a:A, _f_sam:A, _cr_sam:A, _w:A,
                      *eval_opts):

    tp,idx=first_update(population, _ftdiym, f_mem, _f_sam, f_bds, f_scl, cr_mem, _cr_sam, cr_bds, cr_scl, p, pop_eval, *eval_opts)
    a_idx=0
    f_idx=0
    cr_idx=0
    for current_generation in range(max_iters):
        cmn.uchoice_mutator(population, _m_pop, _cr_sam, bounds, reject_mx, cross_apply,
                            cmn.c_to_pbesta_mutate, cfg._C_T_PBA_M_R,_idx, _crr, _t_pop, idx, tp, _f_sam,_a[:a_idx])
        tp,idx,a_idx,f_idx,cr_idx=iter_update_shade(population, _m_pop, _ftdiym, idx, a_idx, _a, _w, f_mem, _f_sam, f_bds, f_scl, f_leh_order, f_idx, cr_mem, _cr_sam, cr_bds, cr_scl, cr_idx, p, pop_eval, monitor, *eval_opts)
        if stop_apply is not None and stop_apply(population,_ftdiym):break

@register_jitable(**cfg.jit_s)
def _lshade_c_t_pbesta(population: A,pop_min:int,p_power_scale:float,
                       bounds: A,
                       reject_mx:int,
                       max_evals:int,
                       f_mem:A, f_bds: A, f_scl:float, f_leh_order:float,
                       cr_mem:A, cr_bds: A, cr_scl:float,  #lets assume no f_init as f_mem should already be initted.
                       p:float|np.ndarray,  #extend p to mem adaption.
                       cross_apply: Callable,
                       stop_apply: Callable,
                       pop_eval: Callable,
                       monitor:Callable,
                       _m_pop:A, _idx:A, _crr:A, _t_pop:A, _ftdiym:A, _a:A, _f_sam:A, _cr_sam:A, _w:A,
                       *eval_opts):
    #tbh this should be renamed rshade, or pshade given the generalized downscaling factor is no longer just order 1, but so user's know the algo
    #they are using, still calling it lshade.
    ogs=population.shape[0]
    #Solved integral estimate of maximum or total iterations, before max_evals is reached, it's only used to scale population and
    #not actually stop iteration because it is an estimate and can be off by ~1% at times, a little more than just rounding to next iteration.
    #A binary search could be a bit more accurate, but I like to use less FLOPs for something trivial.
    max_iters=calc.population_reduction_evaltarget(max_evals,ogs,pop_min,p_power_scale)
    #Unlike other implementations, I let archive be any size for pbesta, so we need to reduce it proportionally to pop_size reduction
    #Which is what a_ params are for
    a_ogs=_a.shape[0]
    a_min=max(cmn.ri64(pop_min*a_ogs/ogs),1)
    ftp=p
    p,tp,idx=first_update(population, _ftdiym, f_mem, _f_sam, f_bds, f_scl, cr_mem, _cr_sam, cr_bds, cr_scl, ftp, pop_eval, *eval_opts)
    t_evals=a_idx=f_idx=cr_idx=itr=0
    while t_evals < max_evals:
        cmn.uchoice_mutator(population, _m_pop, _cr_sam, bounds, reject_mx, cross_apply,
                            cmn.c_to_pbesta_mutate, cfg._C_T_PBA_M_R,_idx, _crr, _t_pop, idx, tp, _f_sam,_a[:a_idx])
        tp,idx,a_idx,f_idx,cr_idx=iter_update_shade(population, _m_pop, _ftdiym, idx, a_idx, _a, _w, f_mem, _f_sam, f_bds, f_scl, f_leh_order, f_idx, cr_mem, _cr_sam, cr_bds, cr_scl, cr_idx, p, pop_eval, monitor, *eval_opts)
        if stop_apply is not None and stop_apply(population,_ftdiym):break
        t_evals+=_m_pop.shape[0]
        tp,population,_a,ogs,pop_min,itr,max_iters,p_power_scale,_m_pop,idx,_idx,_ftdiym,_f_sam,_cr_sam,ftp,p,a_ogs,a_min,a_idx\
        =scale_shade_update(population,_a,ogs,pop_min,itr,max_iters,p_power_scale,_m_pop,idx,_idx,_ftdiym,_f_sam,_cr_sam,ftp,p,tp,a_ogs,a_min,a_idx)
        itr+=1

@register_jitable(**cfg.jit_s)
def _ilshade_c_t_pbesta(population: A,pop_min:int,p_power_scale:float, distance_metric:Callable,
                       bounds: A,
                       reject_mx:int,
                       max_evals:int,
                       f_mem:A, f_bds: A, f_scl:float, f_leh_order:float,
                       cr_mem:A, cr_bds: A, cr_scl:float,  #lets assume no f_init as f_mem should already be initted.
                       p:np.ndarray,  #assume only a float, for population proportional rescaling.
                       cross_apply: Callable,
                       stop_apply: Callable,
                       pop_eval: Callable,
                       monitor:Callable,
                       _m_pop:A, _idx:A, _crr:A, _t_pop:A, _ftdiym:A, _a:A, _f_sam:A, _cr_sam:A, _w:A,
                       *eval_opts):
    ogs=population.shape[0]
    max_iters=calc.population_reduction_evaltarget(max_evals,ogs,pop_min,p_power_scale)
    a_ogs=_a.shape[0]
    a_min=max(cmn.ri64(pop_min*a_ogs/ogs),1)
    p_min=ftp=p[0]
    p_max=p[1]
    p,tp,idx=first_update(population, _ftdiym, f_mem, _f_sam, f_bds, f_scl, cr_mem, _cr_sam, cr_bds, cr_scl, ftp, pop_eval, *eval_opts)
    t_evals=a_idx=f_idx=cr_idx=itr=0
    f_mem[-1]=cr_mem[-1]=.9 #last index of memory is always .9 in ilshade.
    while t_evals < max_evals:
        #ilshade's hardcoded parameter staging, according to cec benchmark, this along with memory .5 ema makes ilshade outperform lshade significantly.
        #Meaning it could be worth exploring this avenue further, since its so simple and easy to optimize as hyperparameters.
        ftp=ilshade_samplepatch(_cr_sam,_f_sam,itr,max_iters,t_evals, max_evals, p_min, p_max)
        cmn.uchoice_mutator(population, _m_pop, _cr_sam, bounds, reject_mx, cross_apply,
                            cmn.c_to_pbesta_mutate, cfg._C_T_PBA_M_R,_idx, _crr, _t_pop, idx, tp, _f_sam,_a[:a_idx])
        tp,idx,a_idx,f_idx,cr_idx=iter_update_ilshade(population, _m_pop, _ftdiym, idx, a_idx, _a, _w, f_mem, _f_sam, f_bds, f_scl, f_leh_order, f_idx, cr_mem, _cr_sam, cr_bds, cr_scl, cr_idx, p, pop_eval, monitor, *eval_opts)
        f_idx%=(f_mem.shape[0]-1) #ilshade low effort hack, the last memory item needs to be .9 for CR and F
        cr_idx%=(cr_mem.shape[0]-1)
        if stop_apply is not None and stop_apply(population,_ftdiym):break
        t_evals+=_m_pop.shape[0]
        tp,population,_a,ogs,pop_min,itr,max_iters,p_power_scale,_m_pop,idx,_idx,_ftdiym,_f_sam,_cr_sam,ftp,p,a_ogs,a_min,a_idx\
        =scale_shade_update(population,_a,ogs,pop_min,itr,max_iters,p_power_scale,_m_pop,idx,_idx,_ftdiym,_f_sam,_cr_sam,ftp,p,tp,a_ogs,a_min,a_idx)
        itr+=1

@nb.njit(**cfg.jit_s)#,inline='always') #would need a pretty large high dim population for cfg.jit_tp
def ilshade_samplepatch(cr_sam,f_sam,itr,max_iters,t_evals,max_evals,p_min,p_max):
    if itr<max_iters*3/4:
        fmax=.7 if itr<max_iters/4 else .8 if itr<max_iters/2 else .9
        for i in range(f_sam.shape[0]):
            f_sam[i]=min(f_sam[i],fmax)
    if itr<max_iters/2:
        crmin=.5 if itr<max_iters/4 else .25
        for i in range(cr_sam.shape[0]):
            cr_sam[i]=max(cr_sam[i],crmin)
    ftp=((p_max-p_min)*t_evals/max_evals) + p_min
    return ftp


@nb.njit(**cfg.jit_s)#,inline='always') #would need a pretty large high dim population for cfg.jit_tp
def scale_shade_update(population,_a,ogs,pop_min,itr,max_iters,p_power_scale,_m_pop,idx,_idx,_ftdiym,_f_sam,_cr_sam,ftp,p,tp,a_ogs,a_min,a_idx):
    n_pz = max(cmn.ri64(ogs + (pop_min - ogs) * ((itr / max_iters) ** p_power_scale)), pop_min)
    if population.shape[0] > n_pz:
        # idx is the sorted best of population
        pz = n_pz
        _m_pop = _m_pop[:pz]
        _m_pop = population[idx[:pz]]
        lp = population[:pz]
        population = _m_pop
        _m_pop = lp  # So we only do one copy into m_pop then turn it into population, while old population becomes m_pop, a bit quicker.
        # change population dependent sizes.
        _idx = _idx[:, :pz]
        _ftdiym = _ftdiym[:pz]
        _f_sam = _f_sam[:pz]
        _cr_sam = _cr_sam[:pz]
        _setip(ftp, pz, p)
        tp = cmn._pset(p)
        # For developer: In the future save original arrays at beginning then subindex into different variables, so that scaling can increase or decrease.
        # Also user now needs to make sure the returned fitness array in eval_opts is also dependent on pop size.
    n_az = max(cmn.ri64(a_ogs + (a_min - a_ogs) * ((itr / max_iters) ** p_power_scale)), a_min)
    if _a.shape[0] > n_az:
        az = n_az
        a_idx = min(a_idx, az)
        _a = _a[:az]
    return tp,population,_a,ogs,pop_min,itr,max_iters,p_power_scale,_m_pop,idx,_idx,_ftdiym,_f_sam,_cr_sam,ftp,p,a_ogs,a_min,a_idx


@register_jitable(**cfg.jit_s)
def first_update(population, _ftdiym, f_mem, f_sam, f_bds, f_scl, cr_mem, cr_sam, cr_bds, cr_scl, ftp, pop_eval, *eval_opts):
    _ftdiym[:] = pop_eval(population, *eval_opts)
    idx = np.argsort(_ftdiym)
    p=_inip(ftp) #in float or float array, out init int or int array
    _setip(ftp,population.shape[0],p)
    tp=_mk_paramsamples(population,f_mem,f_sam,f_bds,f_scl,cr_mem,cr_sam,cr_bds,cr_scl,p)
    return p,tp,idx

@register_jitable(**cfg.jit_s)
def iter_update_shade(population, m_pop, _ftdiym, idx, a_idx, _a, _w, f_mem, f_sam, f_bds, f_scl, f_leh, f_idx, cr_mem, cr_sam, cr_bds, cr_scl, cr_idx, p, pop_eval, monitor, *eval_opts):
    ftdiym = pop_eval(m_pop, *eval_opts)
    f_idx,cr_idx,bdx=shade_memupdate(_ftdiym, ftdiym, idx,_w,f_mem,f_idx,f_sam,f_leh,cr_mem,cr_idx,cr_sam)
    cmn._meval(monitor, population, m_pop, _ftdiym, *eval_opts) #can't be sure this is njit so leaving here.
    restof_update(population,bdx,a_idx,_a,m_pop, _ftdiym, ftdiym,f_mem,f_sam,f_bds,f_scl,cr_mem,cr_sam,cr_bds,cr_scl,p)
    return p,idx,a_idx,f_idx,cr_idx

@register_jitable(**cfg.jit_s)
def iter_update_ilshade(population, m_pop, _ftdiym, idx, a_idx, _a, _w, f_mem, f_sam, f_bds, f_scl, f_leh, f_idx, cr_mem, cr_sam, cr_bds, cr_scl, cr_idx, p, pop_eval, monitor, *eval_opts):
    ftdiym = pop_eval(m_pop, *eval_opts)
    f_idx,cr_idx,bdx=ilshade_memupdate(_ftdiym, ftdiym, idx,_w,f_mem,f_idx,f_sam,f_leh,cr_mem,cr_idx,cr_sam)
    cmn._meval(monitor, population, m_pop, _ftdiym, *eval_opts) #can't be sure this is njit so leaving here.
    restof_update(population,bdx,a_idx,_a,m_pop, _ftdiym, ftdiym,f_mem,f_sam,f_bds,f_scl,cr_mem,cr_sam,cr_bds,cr_scl,p)
    return p,idx,a_idx,f_idx,cr_idx

@nb.njit(**cfg.jit_s)
def shade_memupdate(_ftdiym, ftdiym, idx,_w,f_mem,f_idx,f_sam,f_leh,cr_mem,cr_idx,cr_sam):
    bdx = cmn.select_better(_ftdiym, ftdiym, idx)
    #in theory we could make SHADE memory dependent on recency too.
    bds=bdx.shape[0]
    if bds != 0: #turn these into indexing loops later, for faster compiles and a bit more performance.
        _w[:bds]=_ftdiym[bdx]-ftdiym[bdx] #we are minimizing _ft.. is population fitness, ft.. is mutation fitness
        _w[:bds]/=sum(_w[:bds])
        f_mem[f_idx]= sum(_w[:bds]*(f_sam[bdx] ** f_leh)) / sum(_w[:bds]*(f_sam[bdx] ** (f_leh - 1)))
        f_idx+=1
        f_idx%=f_mem.shape[0]
        cr_mem[cr_idx] = sum(_w[:bds]*cr_sam[bdx]) / bds
        cr_idx += 1
        cr_idx %= cr_mem.shape[0]
    return f_idx,cr_idx,bdx

@nb.njit(**cfg.jit_s)
def ilshade_memupdate(_ftdiym, ftdiym, idx,_w,f_mem,f_idx,f_sam,f_leh,cr_mem,cr_idx,cr_sam):
    bdx = cmn.select_better(_ftdiym, ftdiym, idx)
    bds=bdx.shape[0]
    if bds != 0: #Only difference is that it uses .5 ema smoothing from last generation, keeping memory size the same, hmm...
        _w[:bds]=_ftdiym[bdx]-ftdiym[bdx]
        _w[:bds]/=sum(_w[:bds])
        nxf=sum(_w[:bds]*(f_sam[bdx] ** f_leh)) / sum(_w[:bds]*(f_sam[bdx] ** (f_leh - 1)))
        f_mem[f_idx]=(f_mem[f_idx-1]+nxf)/2 #if _idx 0 then it will average with the .9 at end of array, pretty sure this is desired behavior.
        f_idx+=1
        f_idx%=f_mem.shape[0]
        nxcr=sum(_w[:bds]*cr_sam[bdx]) / bds
        cr_mem[cr_idx] =(cr_mem[cr_idx-1]+nxcr)/2
        cr_idx += 1
        cr_idx %= cr_mem.shape[0]
    return f_idx,cr_idx,bdx

@nb.njit(**cfg.jit_s)
def restof_update(population,bdx,a_idx,_a,m_pop, _ftdiym, ftdiym,f_mem,f_sam,f_bds,f_scl,cr_mem,cr_sam,cr_bds,cr_scl,p):
    a_idx=cmn.update_poparchive(population,bdx,a_idx,_a)
    cmn.place_popfit(population, m_pop, _ftdiym, ftdiym, bdx)
    idx = np.argsort(_ftdiym)  # always needs to be done, idx reusage, if improvements are very rare, might be a little more efficient to have a separate idx array
    p=_mk_paramsamples(population,f_mem,f_sam,f_bds,f_scl,cr_mem,cr_sam,cr_bds,cr_scl,p)
    return a_idx,idx,p

@nb.njit(**cfg.jit_s)
def _mk_paramsamples(population,f_mem,f_sam,f_bds,f_scl,cr_mem,cr_sam,cr_bds,cr_scl,p):
    tp = cmn._pset(p)
    crr=cr_mem.shape[0]
    fr=f_mem.shape[0]
    for i in range(population.shape[0]):  # cauchy and normal
        cr_sam[i] = min(cr_bds[0], max(rand.normalvariate(cr_mem[rand.randrange(0,crr)], cr_scl), cr_bds[1]))
        fg=rand.randrange(0,fr)
        for n in range(8):
            f_sam[i] = f_mem[fg] + m.tan(m.pi * (rand.random() - 0.5)) * f_scl
            if  f_bds[1] > f_sam[i] and f_sam[i] >= f_bds[0]:
                break
        f_sam[i] = max(min(f_sam[i], f_bds[1]), f_bds[0])
    return tp

def _setip(p,pz,pr):
    if type(p) is float:
        return max(cmn.ri64(p * pz), 1)
    else: #its array
        pr[0],pr[1]=max(min(m.ceil(p[0]*pz),pz)-1,1),max(min(m.ceil(p[1]*pz),pz),2)
        return pr
@overload(_setip,**cfg.jit_s)
def _setip_(p,pz,pr):
    if isinstance(p,types.Float):
        _p= lambda p,pz,pr: max(cmn.ri64(p * pz), 1)
    else:  # its array
        def _p(p,pz,pr):
            pr[0], pr[1] = max(min(m.ceil(p[0] * pz), pz) - 1, 1), max(min(m.ceil(p[1] * pz), pz), 2)
            return pr
    return _p

def _inip(p):
    return 0 if type(p) is float else np.empty((2,),dtype=np.int64)
@overload(_inip,**cfg.jit_s)
def _inip_(p):
    if isinstance(p,types.Float): return lambda p: 0
    else: return lambda p: np.empty((2,),dtype=np.int64)