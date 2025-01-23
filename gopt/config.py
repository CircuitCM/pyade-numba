
#NOTE: All lower case variables can be modified to change compile time settings. Upper case leave alone unless you know what you are doing.
#The user can import this module first, change these values like gopt.config.reject_sample_dimorder=2. Then import the optimizer.
#Remember to trigger recompilations/reimport of optimizers if editing these values.

#This might now actually introduce some bias against search on certain dimensions given binomial cross gaurantee not a single dimension for
# reject_sample_dimorder=1. #The order/power scale relationship between the # optimization parameters and # rejection sampling reattempts.
# reject_sample_mult=.25 #The constant multiple of the order scaled # optimizer dimensions to determine re-sampling reattempts.
# reject_sample_mn=2 #Rejection sampling minimum attempts, must be at least 1.
# Rejection sampling example for 50 dims: min(100,max(2,ceil(.2*(50**1.5))))=71
#By default search space does not have boundaries for better search crossover results between generations.

from numba.core.cpu_options import ParallelOptions
import numba as nb
from numba.core.extending import register_jitable,overload

#An extremely hacky method I made to get numba to disable/re-enable parallel dispatch mid-IR processing. This is useful for eg:
#Optimizing the settings/hyperparameters of a DE compiled optimizer pipeline, so that you can run multiple optimizations in parallel each with a low
#population count, instead of less efficiently utilizing parallel recourses and launching more vthreads than there are real threads.
class DBool: #Boolean singleton
    def __init__(self,iv):
        self.v=iv

    def __bool__(self):
        return self.v

    def __repr__(self):
        return f'{self.v}'

class NBPopts(ParallelOptions): #hacked parallel options, needed to get the bool singleton past the type check.

    def __init__(self,value:DBool):
        self.enabled = value
        self.comprehension = value
        self.reduction = value
        self.inplace_binop = value
        self.setitem = value
        self.numpy = value
        self.stencil = value
        self.fusion = value
        self.prange = value

parallel_enabled=True #to enable/disable explicit parallel branches. eg you have something that uses thread_ids instead of just implicit concurrency.
parallel_dec_toggle=NBPopts(DBool(True))
nc_t= nb.njit(lambda:None) #When replaced with a new registered lambda and added as an unused argument to an njit function, it will force a recompilation for that section of the IR chain, maybe.
_dft=dict(fastmath=True, error_model='numpy',cache=False) #numba should sense changes to globals so cache shouldn't interfere
jit_s=_dft|dict(parallel=False) #jit sync only settings. Sync Dispatchers can still call parallel jit funcs, fyi.
jit_tp=_dft|dict(parallel=parallel_dec_toggle) #jit toggle parallel settings. #parallel_dec_toggle
jit_ap=_dft|dict(parallel=True) #jit always parallel settings.

def disable_optional_parallelism():
    global parallel_enabled,parallel_dec_toggle,nc_t
    if parallel_enabled:
        parallel_enabled=False
        parallel_dec_toggle.enabled.v=False
        nc_t =  nb.njit(lambda: None)

def enable_optional_parallelism():
    global parallel_enabled,parallel_dec_toggle,nc_t
    if not parallel_enabled:
        parallel_enabled=True
        parallel_dec_toggle.enabled.v=True
        nc_t =  nb.njit(lambda: None)

#unfortunately doesn't work, can only toggle outside of IR processing eg before the first call that triggers a compilation, then after.
@overload(disable_optional_parallelism, inline='always')
def _dispb():
    disable_optional_parallelism()
    return None

@overload(enable_optional_parallelism, inline='always')
def _enpb():
    enable_optional_parallelism()
    return None




#private module vars for comparisons in jitted functions.
_M_BINARY=0
_M_CUR_T_BEST2=1
_M_CUR_T_PBEST=2
_M_CUR_T_RAND1=3
_M_CUR_T_PBESTW=4
_M_CUR_T_PBESTA=5

#Population selection for different mutation strategies. Don't change unless rewriting mutation operators.
_BIN_M_R=3
_C_T_B2_M_R=2
_C_T_PB_M_R=2
_C_T_R1_M_R=3
_C_T_PBW_M_R=2
_C_T_PBA_M_R=2

#Classes for user selection in py
class MutationSelector:
    BINARY=_M_BINARY
    CUR_T_BEST2=_M_CUR_T_BEST2
    CUR_T_PBEST=_M_CUR_T_PBEST
    CUR_T_RAND1=_M_CUR_T_RAND1
    CUR_T_PBESTW=_M_CUR_T_PBESTW
    CUR_T_PBESTA=_M_CUR_T_PBESTA

class CrossoverSelector:
    BIN = 0 #only bin is available for now.
    EXP = 1

class CallStrategy:
    RUN=0 #will simply try to run, such that apply_de is the executor. Will first attempt nopython compilation, will fallback to python mode.
    OBJ=1 #will return a runnable python object with all parameters that wouldn't necessarily need a recompilation if changed.
    OBJ_JIT=2 #Same as OBJ but briefly runs it first for two generations to trigger a compilation, if your fitness function takes forever to evaluate then use previous mode.
    RAW=3 #For advanced users to construct their own jitted routines. The raw implementations and args


#DEPRICATED, experiment with later.
#Sub policies for best-of-n resampling, performed before fitness function eval. As it's a randomized rejection sampling method it can be used
#on any DE algorithm without altering the deterministic portion of the mutation+crossover strategy. Used to counteract optimizer biases,
#reduce evaluation count and increase solution quality.
class SamplerRejectionPolicy: #well rip.
    NONE = 0
    REJECT_BOUNDS= 1 #Will abbreviate to B
    NORM_CLONE=2 #to C
    NORM_DIM_DEGEN=3 #to D
    B_C=4
    B_D=5
    C_D=6
    B_C_D=7

#Reject norm duplicates:
#Either sort by the largest difference!=current and make a proximity metric, but that's at least m*nlogn, not my favorite.
#Or make a metric that uses a strongly dominant proximity cost, so it's m*n, eg some kind of softmax or logistic scheme.
#Actually it remains O(m*n) if you track the max while iterating over the proximities for the dominance metric, if you want
#you can track more than one like largest 1 or n. Problem is with max 1, if a vector lands near two already close points equally, that
#could end up being worth only 1/2 of what landing close to one point is worth, maybe there is some sense to it though idk. Think about
#replacing with another dominance metric that is still O(m*n). It is argmax(1/max((x_ibs - x_mutebs),1e-8))/sum(1/max((x_ibs - x_mutebs),1e-8)).
#Reject norm tangency loss:
#There are many explanations regarding why DE algorithms may fail to find the true global optima on a high dimensional search space
#when it's low dimensional counterpart succeed with greater frequency.
#They include: CR too high, population count low, non-separability, dominant search parameters.
#However, all of these issues contribute to a particular geometric challenge; the loss of tangency along dimensions of a search space.
#Or what could be called dimensional collapse, when the bounds normalized differences between population members along an axis
#is orders of magnitude smaller than the other dimensions. Once this occurs, it becomes significantly more challenging for the
#optimizer to break out of localities along that axis as there are no longer any differences left that could encourage it.
#It may also be the primary reason it is challenging to achieve global convergence with a population count < total # dimensions.
#Reject Tangency Norm Loss addresses this by selecting the sample out of n samples that results in the smallest result for:
#std(x bound-scaled)/abs_diff(x bound-scaled)-0.7071067812, this is an expensive calculation when the population count is high,
#per individual O(m*n) n=#individuals m=# axes. And it's still not perfect when multiple new population members are accepted,
#but it's not far off. As my library has the expectation of a significantly more expensive search space evaluation its a non-issue for me.
#An alternative O(m) would be L2norm(x_mutebs - x_meanbs)/L1norm(x_mutebs - x_meanbs)-0.7071067812, when scale up to many parameters,
#The deviation between these metrics is likely to be less significant. maybe. Actually check.
#Later change your terms to "degenerate hyperplane/dimension"
#Also you can potentially flip this metric to intentionally increase sparcity... maybe not actually idk.
#according to gpt-01 to actually measure how spherical it will be you need the parameter covariance matrix.
#You were completely wrong about the math but I think your idea is still there, rewrite this.
#Yeah, the mean version really might make more sense, but not sure try later. while current to all dim efficiency will encourage it
#to clump on the diagonals. at least mean to all encourages polaristic/symmetric diagonals
#You can consider trying to make CMA-JADE by removing the ES from ESCMA and adding jade, using CMA to bias the rejection sampling.
#(Failed experiment for now, consider the ideas below instead if you want to give it another try).

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

#Implementation Schedule:
#Jade implementation d, setup n
#Sade implementation d, setup n
#Shade implementation n, setup n
#LShade implementation n, setup n
#ILShade implementation n, setup n
#lshadecnepsin implementation n, setup n
#mpede - maybe
#jso -maybe
#ADEGTO from that book, sb easy ish, implementation n, setup n

#After essential implementations:
#Jade with archive, IJADE maybe, see that 2013 paper, but prob not worth it, sal-shade-epsin and the others before it likely are tho.
#Own implementation: JADE-Sep (maybe some other IJADE features), F is now separable by dimension so it's a 2D sample and 1D for U_f,
#CR can also be seperable actually, it might be a decent idea, letting certain dimensions adapt first, however could introduce sign bias.
#CR technically doesn't need hard limits either, as long as 1 sample is forced.
#CoDE
#current-to-pbest2,
#current-to-pbesta,
#current-to-pbest2a, a is archive, 2 includes either another best difference or random diff.

#anti-dim collapse: argmin/sample-min population's std(x bound-scaled)/absdev(x bound-scaled)-0.7071067812, such that the cost will vary between (0,~.292]
#For proximity you'll want something else... like some kind of exponential or L2 version (failed to improve results for now).

#Rejection sampling for different biases, such as proximity to other member that are not it's current, dimensional flattening avoidance. (failed to improve results).

#https://link.springer.com/article/10.1007/s42979-024-03062-2 maybe.

#https://en.wikipedia.org/wiki/Evolutionary_multimodal_optimization basically finding many local solutions.