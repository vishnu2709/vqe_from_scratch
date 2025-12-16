# vqe_from_scratch
Trying to implement variational quantum eigensolver without using any quantum computing libraries (qiskit, openfermion, PennyLane etc). Aimed so as to understand VQE in the greatest possible depth. PySCF has been used for the "classical" quantum chemistry.

Currently the code implemented (vqe.py) is very simple, and has two possible ansatz that can be used for H2, represented with 4-qubits without any symmetry reduction. Both are terrible (one returns RHF result, the other shows dissociation but optimizer isn't giving good results near equilibrium bond length. Improvements under progress).

test.py calls vqe.py routines to generate disassociation curve for H2.
