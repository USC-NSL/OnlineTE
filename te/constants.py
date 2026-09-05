import multiprocessing
from gurobipy import GRB


TM_DIR = "traffic_models/traffic_matrices"

FLOAT_RES = 1e-6
"""When deciding feasibility, if two values are closer than this, we consider them to be the same"""

MINIMUM_NORM = 1e-12
"""This value is the smallest norm a vector can have while not being considered just 0"""

# GUROBI SPECIFIC

DEFAULT_SOLVER_METHOD = GRB.METHOD_BARRIER
"""We _ALWAYS_ use Barrier method. It is the only one that is fast enough for very large LPs"""

DEFAULT_CROSSOVER = 0
"""
We don't need basic solutions per-se, thus, we will disable crossover to make things fair for Gurobi
When generating baseline solutions, one shortcut we can use is to use Barrier instead of Simplex, but
with crossover explicitly enabled. This gives us a solution much faster.
"""

DEFAULT_OPTIMALITY_TOLERANCE = 1e-3
"""
Solutions don't need to be too accurate, since we have to simplify them later anyway.
Thus, we relax barrier convergence tolerance, and stop within 1 percent of the optimal.
In Gurobi, `BarConvTol` controls this, and `OptimalityTol` does the same for simplex
algorithms. In order to prevent accidentally using different tolerances for them, we
will only use a single parameter and set both of the tolerances to the same value.
"""

DEFAULT_FEASIBILITY_TOLERANCE = 1e-4
"""
This affects flow conservation and demand constraints.
We need this to be quite tight, so this we keep smaller.
"""

DEFAULT_NUMERIC_FOCUS = 1
"""
For simplex on a very large topology, it is quite common to run into numerical issues.
For such cases, this value should probably be 2 instead
"""

DEFAULT_PRESOLVE = 0
"""
Presolve hurts Barrier at large scale as far as we could see in our tests.
It helps Simplex quite a bit though
"""

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


# Distributed implementation specific
DEFAULT_RPC_PORT = 13000
"""
Can be used to bind RPC servers when spawning a lone node on just one
independent machine.
"""

DEFAULT_SCATTER_ADDRESS = "224.0.0.10"
"""
Default multicast group address used for scattering updates from the controller
to many workers.
"""

DEFAULT_SCATTER_PORT = 12000
"""
Default UDP port used to bind for IP multicasting.
"""

SHOW_PROGRESS_BAR = True
"""
Print `tqdm` progress bars
"""

"""
Values for vector consensus tests.
"""

SEVERE_CONSENSUS_VIOLATION_REL_TOL = 5e-2
"""
If any element within two vectors are this far apart relateively,
then they are not in consensus at all.
"""


NUM_PROCS = multiprocessing.cpu_count()
"""Number of available cores"""

MAX_NUMBER_OF_SINGLE_HOST_WORKERS = min(24, max(NUM_PROCS - 4, 1))
"""
Maximum number of processes that will be spawned to do any parallel task
on a single host.
We avoid going up to exactly the CPU count, since that will bring a lot of
contention and cause problems.
We leave 4 cores alone at all times for other things.
"""

MAX_NUMBER_OF_COMMODITIES_PER_CORE = 5000
"""
For operations done on a single host that can be parallelized over commodities,
this is the maximum number of commodities that we will handle with one core.
This number will be ignored in case the compute is limited and we are oversubscribed.
"""

GRPC_ARRAY_STREAM_MAX_LEN = 2**20
"""
Maximum length of an array to be transfered at once over a gRPC stream.
"""