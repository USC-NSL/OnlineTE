"""
Some toy code for working with asynchronous ADMM solvers for Consensus/Sharing 
problems.

Much of this is based on the following:
    https://proceedings.mlr.press/v32/zhange14.pdf#page=9&zoom=100,0,0
"""


import time
import tqdm
import queue
import gurobipy
import threading
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Set


NUM_WORKERS = 32

MASTER_QUEUE = queue.SimpleQueue()
WORKER_QUEUES = [queue.SimpleQueue() for _ in range(NUM_WORKERS)]

IS_ALIVE = True


def handler(*args):
    global IS_ALIVE
    IS_ALIVE = False
    print("INT!")


def cleanup():
    global IS_ALIVE
    global MASTER_QUEUE
    global WORKER_QUEUES

    IS_ALIVE = True
    MASTER_QUEUE = queue.SimpleQueue()
    WORKER_QUEUES = [queue.SimpleQueue() for _ in range(NUM_WORKERS)]


class AsynchronousConsensusMaster:
    def __init__(self, n: int, dim: int, maxiter: int, param_K: int, param_Tau: int, theta: np.ndarray):
        assert param_Tau >= 1 and param_K <= n
        self._num_workers = n
        self._dim = dim
        self._maxiter = maxiter
        self._param_K = param_K
        self._param_Tau = param_Tau
        self._theta = theta
        self._clock: int = 0
        """Local clock"""
        self._timer: List[int] = [0 for _ in range(n)]
        """Time of last worker arrival"""
        self._X: np.ndarray = np.zeros(shape=(n, dim))
        self._r: np.ndarray = np.zeros(shape=(n, dim))
        self._Z: np.ndarray = np.zeros(shape=(dim,))
        self._worker_updates: List[Tuple[int, np.ndarray, np.ndarray]] = []
        """List of arrived updates"""
        self._arrivals: Set[int] = set()
        """Set of arrived worker IDs"""
        self._obj: List[float] = []

    def can_update(self) -> bool:
        """
        Returns `True` when we are ready to commit and scatter an update.
        This happens if both of the following are true:
            - The number of arrived workers is at least `K`
            - No worker update is older than `Tau`
        """
        return len(self._arrivals) >= self._param_K and max(self._timer) < self._param_Tau
    
    def update_Z(self):
        for worker, new_X, new_r in self._worker_updates:
            self._X[worker, :] = new_X
            self._r[worker, :] = new_r
        self._Z = np.mean(self._X + self._r, axis=0)
    
    def notify_workers_and_reset(self):
        for i, _, _ in self._worker_updates:
            WORKER_QUEUES[i].put_nowait(self._Z)
        self._worker_updates.clear()
        self._arrivals.clear()
    
    def wait_until_update(self) -> bool:
        while IS_ALIVE:
            try:
                item = MASTER_QUEUE.get(timeout=1.0)
                worker_id: int = item[0]
                self._arrivals.add(worker_id)
                self._worker_updates.append(item)
                self._timer[worker_id] = 0
            except queue.Empty:
                pass
            finally:
                if self.can_update():
                    return True
        return False
    
    def increment_timer(self):
        for i in set(range(self._num_workers)).difference(self._arrivals):
            self._timer[i] += 1
    
    def increment_clock(self):
        self._clock += 1
    
    def record_objective(self):
        self._obj.append(np.sum(0.5 * np.linalg.norm(self._Z - self._theta, axis=1)**2))
    
    def main_loop(self):
        for _ in tqdm.tqdm(range(self._maxiter)):
            if not self.wait_until_update():
                break
            self.increment_timer()
            self.update_Z()
            self.notify_workers_and_reset()
            self.record_objective()
            self.increment_clock()
        global IS_ALIVE
        IS_ALIVE = False
    

class AsynchronousConsensusWorker:
    def __init__(self, worker_id: int, theta: np.ndarray, rho: float, delay: int):
        assert np.ndim(theta) == 1
        self._worker_id = worker_id
        self._delay = delay
        self._theta = theta
        self._rho = rho
        self._dim = theta.shape[0]
        self._clock: int = 0
        self._X: np.ndarray = np.zeros(shape=(self._dim,))
        self._r: np.ndarray = np.zeros(shape=(self._dim,))
        self._Z: np.ndarray = np.zeros(shape=(self._dim,))

    def update_X(self):
        self._X = (self._theta + self._rho * (self._Z - self._r)) / (1 + self._rho)
    
    def increment_clock(self):
        self._clock += 1
    
    def update_r(self):
        self._r += self._X - self._Z
    
    def notify_master(self):
        if self._delay > 0:
            time.sleep(self._delay / 1000)
        MASTER_QUEUE.put_nowait((self._worker_id, self._X, self._r))
    
    def wait_for_update(self) -> Optional[np.ndarray]:
        while IS_ALIVE:
            try:
                item = WORKER_QUEUES[self._worker_id].get(timeout=1.0)
                assert isinstance(item, np.ndarray)
                return item
            except queue.Empty:
                pass
        return None
    
    def main_loop(self):
        while IS_ALIVE:
            self.update_X()
            self.notify_master()
            new_Z = self.wait_for_update()
            if new_Z is not None:
                self._Z = new_Z
                self.update_r()
                self.increment_clock()
            else:
                break


class AsynchronousReserverMaster:
    def __init__(self, partitions: List[int], maxiter: int, param_K: int, param_Tau: int, param_Rho: float, theta: np.ndarray):
        n = len(partitions)
        assert param_Tau >= 1 and param_K <= n
        assert np.ndim(theta) == 1
        dim = theta.shape[0]
        assert sum(partitions) == dim
        self._dim = dim
        self._partitions = param_Tau
        self._num_workers = n
        self._maxiter = maxiter
        self._param_K = param_K
        self._param_Tau = param_Tau
        self._param_Rho = param_Rho
        self._theta = theta
        self._clock: int = 0
        """Local clock"""
        self._timer: List[int] = [0 for _ in range(n)]
        """Time of last worker arrival"""
        self._X: List[np.ndarray] = [np.zeros(shape=(d,)) for d in partitions]
        self._r: List[float] = [0.0 for _ in range(self._num_workers)]
        self._Z: List[float] = [0.0 for _ in range(self._num_workers)]
        self._U: float = 0.0
        self._worker_updates: List[Tuple[int, np.ndarray, float]] = []
        """List of arrived updates"""
        self._arrivals: Set[int] = set()
        """Set of arrived worker IDs"""
        self._obj: List[float] = []
        self._env = gurobipy.Env()
        self._env.start()

    def can_update(self) -> bool:
        """
        Returns `True` when we are ready to commit and scatter an update.
        This happens if both of the following are true:
            - The number of arrived workers is at least `K`
            - No worker update is older than `Tau`
        """
        return len(self._arrivals) >= self._param_K and max(self._timer) < self._param_Tau
    
    def update_Z(self):
        for worker, new_X, new_r in self._worker_updates:
            self._X[worker] = new_X
            self._r[worker] = new_r
        
        bias = np.array([np.sum(x) for x in self._X]) + np.array(self._r)
        RHO = self._param_Rho
        model = gurobipy.Model('reserver', self._env)
        model.Params.OutputFlag = 0
        model.Params.LogFile = ''
        u = model.addVar(lb=0, ub=1, name='u')
        z = model.addVars(self._num_workers, lb=0.0, name='z')
        obj = gurobipy.QuadExpr()
        obj.addTerms(1.0, u)
        for i in range(self._num_workers):
            obj.addTerms(RHO/2, z[i], z[i])
            obj.addTerms(-RHO*bias[i], z[i])
        model.addConstr(z.sum() <= u * self._dim)
        model.setObjective(obj, gurobipy.GRB.MINIMIZE)
        model.optimize()
        self._Z = np.array([z[i].X for i in range(self._num_workers)])
        self._U = u.X
    
    def notify_workers_and_reset(self):
        for i, _, _ in self._worker_updates:
            WORKER_QUEUES[i].put_nowait(self._Z)
        self._worker_updates.clear()
        self._arrivals.clear()
    
    def wait_until_update(self) -> bool:
        while IS_ALIVE:
            try:
                item = MASTER_QUEUE.get(timeout=1.0)
                worker_id: int = item[0]
                self._arrivals.add(worker_id)
                self._worker_updates.append(item)
                self._timer[worker_id] = 0
            except queue.Empty:
                pass
            finally:
                if self.can_update():
                    return True
        return False
    
    def increment_timer(self):
        for i in set(range(self._num_workers)).difference(self._arrivals):
            self._timer[i] += 1
    
    def increment_clock(self):
        self._clock += 1
    
    def record_objective(self):
        self._obj.append(self._U)
    
    def main_loop(self):
        for _ in tqdm.tqdm(range(self._maxiter)):
            if not self.wait_until_update():
                break
            self.increment_timer()
            self.update_Z()
            self.notify_workers_and_reset()
            self.record_objective()
            self.increment_clock()
        global IS_ALIVE
        IS_ALIVE = False
    
    def __del__(self):
        if getattr(self, '_env', None) is not None:
            self._env.close()


class AsynchronousReserverWorker:
    def __init__(self, worker_id: int, partitions: List[int], theta: np.ndarray, delay: int):
        assert np.ndim(theta) == 1
        self._worker_id = worker_id
        assert worker_id < len(partitions)
        self._partitions = partitions
        self._delay = delay
        self._theta = theta
        self._dim = theta.shape[0]
        self._clock: int = 0
        self._X: np.ndarray = np.zeros(shape=(partitions[worker_id],))
        self._r: float = 0.0
        self._Z: float = 0.0
        self._partition_length = partitions[worker_id]
        self._indices = np.cumulative_sum(partitions)

    def update_X(self):
        start = self._indices[self._worker_id] - self._partition_length
        self._X = self._theta[start: start+self._partition_length]
    
    def increment_clock(self):
        self._clock += 1
    
    def update_r(self):
        self._r += np.sum(self._X) - self._Z
    
    def notify_master(self):
        if self._delay > 0:
            time.sleep(self._delay / 1000)
        MASTER_QUEUE.put_nowait((self._worker_id, self._X, self._r))
    
    def wait_for_update(self) -> Optional[np.ndarray]:
        while IS_ALIVE:
            try:
                item = WORKER_QUEUES[self._worker_id].get(timeout=1.0)
                assert isinstance(item, np.ndarray)
                return item
            except queue.Empty:
                pass
        return None
    
    def main_loop(self):
        while IS_ALIVE:
            self.update_X()
            self.notify_master()
            new_Z = self.wait_for_update()
            if new_Z is not None:
                self._Z = new_Z[self._worker_id]
                self.update_r()
                self.increment_clock()
            else:
                break


def consensus_test(K, TAU):
    """
    Unconstrained Consensus is the problem of the form:
        minimize sum_i f_i(x_i)
            s.t. for all i # j: x_i = x_j 
    The simplest problem for this can be of the form:
        minimize sum_i (x_i - theta_i)^2
            s.t. for all i # j: x_i = x_j 
    Where `theta` is a an unknown parameter, and the optimal value
    is the mean of `theta`.

    The ADMM steps to solve this would take the form:
        1) minimize   1/2 (x_i - theta_i)^2 + rho/2 (x_i - z + r_i)^2
        2) minimize rho/2 sum_i (x_i - z + r_i)^2
        3) r_i <- r_i + (x_i - z)
    Which simplifies to:
        1) x_i <- (theta_i + rho * (z - r_i)) / (1 + rho)
        2) z <- mean(x) + mean(r)
        3) r_i <- r_i + (x_i - z)
    We solve a two dimenssional version of this, so each `x_i` and `z`
    is a vector.
    """
    global IS_ALIVE
    cleanup()
    SEED = 12345
    N = 128
    MAX_ITER = 100
    RHO = 1.9

    RNG = np.random.default_rng(seed=SEED)

    theta = RNG.random((NUM_WORKERS, N))
    master = AsynchronousConsensusMaster(NUM_WORKERS, N, MAX_ITER, K, TAU, theta)
    workers = [AsynchronousConsensusWorker(i, theta[i, :], RHO, i*1) for i in range(NUM_WORKERS)]
    worker_threads = [threading.Thread(target=worker.main_loop) for worker in workers]
    master_thread = threading.Thread(target=master.main_loop)
    master_thread.start()
    for t in worker_threads:
        t.start()

    while IS_ALIVE:
        try:
            time.sleep(1.0)
        except KeyboardInterrupt:
            IS_ALIVE = False

    master_thread.join()
    for t in worker_threads:
        t.join()
    
    optimal = 0.5 * np.linalg.norm(theta - np.mean(theta, axis=0)) ** 2
    iterates = np.log(master._obj - optimal)
    iterates = np.clip(np.clip(0.5 * (iterates/np.max(iterates)), a_min=0, a_max=None) + 0.554, a_min=None, a_max=1)

    return iterates


def reserver_test(K, TAU, RHO):
    """
    A variation of the consensus test.
    Here, we change up the problem as follows:
        minimize u + sum_i (x_i - theta_i)^2
            s.t. sum_i x_i <= u.N
                 0 <= u <= 1
                 0 <= x_i
    
    """
    global IS_ALIVE
    cleanup()
    SEED = 12345
    N = 128
    MAX_ITER = 100

    RNG = np.random.default_rng(seed=SEED)

    part_len = N // NUM_WORKERS
    part_rem = N % NUM_WORKERS
    partitions = [part_len for _ in range(NUM_WORKERS)]
    partitions[-1] += part_rem

    theta = RNG.random((N,))
    try:
        master = AsynchronousReserverMaster(partitions, MAX_ITER, K, TAU, RHO, theta)
        workers = [AsynchronousReserverWorker(i, partitions, theta, i*1) for i in range(NUM_WORKERS)]
        worker_threads = [threading.Thread(target=worker.main_loop) for worker in workers]
        master_thread = threading.Thread(target=master.main_loop)
        master_thread.start()
        for t in worker_threads:
            t.start()

        while IS_ALIVE:
            try:
                time.sleep(1.0)
            except KeyboardInterrupt:
                IS_ALIVE = False

        master_thread.join()
        for t in worker_threads:
            t.join()
        
        optimal = theta.sum() / N
        iterates = np.log(abs(master._obj - optimal) + 1e-16)
        # iterates = np.clip(np.clip(0.5 * (iterates/np.max(iterates)), a_min=0, a_max=None) + 0.554, a_min=None, a_max=1)

        return iterates
    except:
        del master
        raise RuntimeError


if __name__ == '__main__':
    # iter1 = consensus_test(8, 1)
    # iter2 = consensus_test(8, 2)
    # iter4 = consensus_test(8, 4)
    # iter8 = consensus_test(8, 8)
    # iter16 = consensus_test(8, 16)
    # plt.plot(iter1)
    # plt.plot(iter2)
    # plt.plot(iter4)
    # plt.plot(iter8)
    # plt.plot(iter16)
    # plt.show()
    iter1 = reserver_test(1, 10000, 1.0)
    plt.plot(iter1)
    plt.show()
