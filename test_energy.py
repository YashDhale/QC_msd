import sys
import os
sys.path.append("/home/yhwach/quantum-msd-tool-main")

from qaoa_engine import build_qubo, qubo_to_ising, _build_energy_vector, _expectation

import numpy as np

k_values = [500, 1000, 2000, 4000]
c_values = [5, 10, 20, 40]

m = 1.0
F0 = 1.0
lam = 4.5

Q, cost_norm = build_qubo(k_values, c_values, m, F0, lam)
h, J, offset = qubo_to_ising(Q)

n_qubits = len(k_values) + len(c_values)
energies = _build_energy_vector(n_qubits, h, J)

print(f"offset for Ising: {offset}")

# Let's decode energies for some bitstrings
def get_energy(bits):
    e = 0
    idx = 0
    for i, b in enumerate(bits):
        idx |= (b << (n_qubits - 1 - i))
    return energies[idx]

print("All zeros (00000000):", get_energy([0]*8))
print("One hot (00010001):", get_energy([0,0,0,1, 0,0,0,1]))
print("All ones (11111111):", get_energy([1]*8))

idx_and_e = list(enumerate(energies))
idx_and_e.sort(key=lambda x: x[1])

print(f"Top 5 lowest states:")
for i in range(5):
    idx, e = idx_and_e[i]
    bits = [(idx >> (n_qubits - 1 - b)) & 1 for b in range(n_qubits)]
    print(bits, e, "True energy =", e + offset + 2*lam)
