
#NOTE: All lower case variables can be modified to change compile time settings. Upper case leave alone unless you know what you are doing.
#The user can import this module first, change these values like pyade.config.reject_sample_dimorder=2. Then import the optimizer.
#Remember to trigger recompilations/reimport of optimizers if editing these values.

reject_sample_dimorder=1. #The order/power scale relationship between the # optimization parameters and # rejection sampling reattempts.
reject_sample_mult=.25 #The constant multiple of the order scaled # optimizer dimensions to determine re-sampling reattempts.
reject_sample_mx=25 #Rejection sampling hard ceiling for any number of dimensions.
reject_sample_min=2 #Rejection sampling minimum attempts, must be at least 1.
# Rejection sampling example for 50 dims: max(2,min(100,ceil(.2*(50**1.5))))=71, 200 dim: max(2,min(100,ceil(.2*(200**1.5))))=100
#By default search space does not have boundaries for better search crossover results between generations.
force_search_bounds=False

#Population selection for different mutation strategies. Don't change unless rewriting mutation operators.
BIN_MUTATE_R=3