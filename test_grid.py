import sys
sys.path.append("/home/yhwach/quantum-msd-tool-main")

from qaoa_engine import build_qubo, qubo_to_ising, _build_energy_vector, _expectation, _qaoa_state
import numpy as np

k_values = [500, 1000, 2000, 4000]
c_values = [5, 10, 20, 40]
m, F0, lam = 1.0, 1.0, 4.5

Q, cost_norm = build_qubo(k_values, c_values, m, F0, lam)
h, J, offset = qubo_to_ising(Q)
n_qubits = len(k_values) + len(c_values)
energies = _build_energy_vector(n_qubits, h, J)

# 1. Grid search for p=1
best_exp = np.inf
best_params = None

gamma_grid = np.linspace(0, 2*np.pi, 30)
beta_grid = np.linspace(0, np.pi, 30)

for g in gamma_grid:
    for b in beta_grid:
        exp = _expectation([g, b], n_qubits, energies, 1)
        if exp < best_exp:
            best_exp = exp
            best_params = [g, b]

print(f"Grid search best exp: {best_exp} at gamma={best_params[0]:.4f}, beta={best_params[1]:.4f}")

# Look at states
state = _qaoa_state(best_params, n_qubits, energies, 1)
probs = np.abs(state)**2

prob_list = []
for idx in range(len(probs)):
    bits = [(idx >> (n_qubits - 1 - b)) & 1 for b in range(n_qubits)]
    bs = "".join(str(b) for b in bits)
    prob_list.append((bs, probs[idx], energies[idx]))

prob_list.sort(key=lambda x: x[1], reverse=True)
print("Top 10 bitstrings from Grid Search:")
for i in range(10):
    print(f"{prob_list[i][0]} : {prob_list[i][1]:.4f} (Ising E={prob_list[i][2]:.4f})")

