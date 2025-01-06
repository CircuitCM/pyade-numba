
#NOTE: All lower case variables can be modified to change compile time settings. Upper case leave alone unless you know what you are doing.
#The user can import this module first, change these values like pyade.config.reject_sample_dimorder=2. Then import the optimizer.
#Remember to trigger recompilations/reimport of optimizers if editing these values.

#This might now actually introduce some bias against search on certain dimensions given binomial cross gaurantee not a single dimension for
# reject_sample_dimorder=1. #The order/power scale relationship between the # optimization parameters and # rejection sampling reattempts.
# reject_sample_mult=.25 #The constant multiple of the order scaled # optimizer dimensions to determine re-sampling reattempts.
# reject_sample_mn=2 #Rejection sampling minimum attempts, must be at least 1.
# Rejection sampling example for 50 dims: min(100,max(2,ceil(.2*(50**1.5))))=71
#By default search space does not have boundaries for better search crossover results between generations.
force_search_bounds=False

#Population selection for different mutation strategies. Don't change unless rewriting mutation operators.
BIN_MUTATE_R=3
C_T_B2_MUTATE_R=2
C_T_PB_MUTATE_R=2
C_T_R1_MUTATE_R=3
C_T_PBW_MUTATE_R=2

class MutationSelector:
    #Last is either BIN : binary cross, EXP: exponential crossover.
    BINARY=0
    CUR_T_BEST2=1
    CUR_T_PBEST=2
    CUR_T_RAND1=3
    CUR_T_PBESTW=4


class CrossoverSelector:
    BIN = 0
    EXP = 1