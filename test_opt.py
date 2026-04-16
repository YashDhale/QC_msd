import sys
import time
sys.path.append("/home/yhwach/quantum-msd-tool-main")

from qaoa_engine import build_qubo, qubo_to_ising, _build_energy_vector, _expectation
from scipy.optimize import differential_evolution, shgo
import numpy as np

k_values = [500, 1000, 2000, 4000]
c_values = [5, 10, 20, 40]
m, F0, lam = 1.0, 1.0, 4.5

Q, cost_norm = build_qubo(k_values, c_values, m, F0, lam)
h, J, offset = qubo_to_ising(Q)
n_qubits = len(k_values) + len(c_values)
energies = _build_energy_vector(n_qubits, h, J)

def objective(x):
    return _expectation(x, n_qubits, energies, 1)

bounds = [(0, 2*np.pi), (0, np.pi)]

# DE
t0 = time.time()
res_de = differential_evolution(objective, bounds, popsize=15, mutation=(0.5, 1), recombination=0.7, tol=1e-3, seed=42)
t1 = time.time()
print(f"DE: {res_de.fun:.4f} in {t1-t0:.4f}s")
print(f"Params: {res_de.x}")

# SHGO
t0 = time.time()
res_sh = shgo(objective, bounds, iters=4)
t1 = time.time()
print(f"SHGO: {res_sh.fun:.4f} in {t1-t0:.4f}s")
print(f"Params: {res_sh.x}")

