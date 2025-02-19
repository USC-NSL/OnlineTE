from gurobipy import GRB

TM_DIR = "traffic_models/traffic_matrices"

"""When deciding feasibility, if two values are closer than this, we consider them to be the same"""
FLOAT_RES = 1e-6

# GUROBI SPECIFIC

# We _ALWAYS_ use Barrier method. It is the only one that is fast enough for very large LPs
DEFAULT_SOLVER_METHOD = GRB.METHOD_BARRIER
# We don't need basic solutions per-se, thus, we will disable crossover to make things fair for Gurobi
DEFAULT_CROSSOVER = 0
# Solutions don't need to be too accurate, since we have to simplify them later anyway.
# Thus, we relax barrier convergence tolerance, and stop within 1 percent of the optimal.
DEFAULT_BARRIER_CONVERGENCE_TOLERANCE = 1e-2
# This affects flow conservation and demand constraints.
# We need this to be quite tight, so this we keep smaller.
DEFAULT_FEASIBILITY_TOLERANCE = 1e-4
DEFAULT_NUMERIC_FOCUS = 1
DEFAULT_GUROBI_LOG_FILE = ''

# DISTRIBUTED SPECIFIC
DEFAULT_EPSILON_OE = 1
DEFAULT_EPSILON_KE = 1
DEFAULT_ALPHA = 5e-2
DEFAULT_BETA = 5e-2
DEFAULT_SEED = 12345
DEFAULT_NUMBER_OF_NODE_PROCESSES = 5

# ADMM SPECIFIC
DEFAULT_RHO = 1e-3
DEFAULT_ETA = 1e-3
DEFAULT_NUMBER_OF_NETWORK_UPDATES = 10
DEFAULT_MU = 10
DEFAULT_TAU_INC = 2
DEFAULT_TAU_DEC = 2
DEFAULT_BIG_GAMMA = 1e-4
DEFAULT_BIG_THETA = 1e-2
