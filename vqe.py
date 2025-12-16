# My attempt at VQE, here we go
# First step, import numpy and pyscf
import numpy as np
from pyscf import gto, scf, ao2mo, fci
import functools as ft
from scipy.optimize import minimize_scalar, minimize


# function that implements the entire quantum circuit
# the input taken is the one parameter of the circuit that needs to be optimized
def generate_state_vector(theta):
   # Let's initialize the four qubits that we need to |0> state
   q_0 = np.array([1,0], dtype=complex)
   q_1 = np.array([1,0], dtype=complex)
   q_2 = np.array([1,0], dtype=complex)
   q_3 = np.array([1,0], dtype=complex)

   # Define the Ry matrix
   Ry = np.array([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

   # Apply Ry to q_1
   q_1_rotated = Ry @ q_1


   # Construct the 2-qubit kron product q_1_rotated x q_0
   q_10 = np.kron(q_1_rotated, q_0)

   # Define the CNOT gate
   cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

   # Apply the CNOT gate to entangle the two qubits
   q_10_entangled = cnot @ q_10

   # Do the kronecker product of the other two qubits
   q_32 = np.kron(q_3, q_2)

   # Take the final kron product to get the final statevector
   q_3210 = np.kron(q_32, q_10_entangled)

   return q_3210

def generate_state_vector_advanced(theta):
   # Let's initialize the four qubits that we need to |0> state
   q_0 = np.array([1,0], dtype=complex)
   q_1 = np.array([1,0], dtype=complex)
   q_2 = np.array([1,0], dtype=complex)
   q_3 = np.array([1,0], dtype=complex)

   theta1, theta2, theta3 = theta

   # Define the Ry matrix
   Ry1 = np.array([[np.cos(theta1/2), -np.sin(theta1/2)],[np.sin(theta1/2), np.cos(theta1/2)]], dtype=complex)
   Ry2 = np.array([[np.cos(theta2/2), -np.sin(theta2/2)],[np.sin(theta2/2), np.cos(theta2/2)]], dtype=complex)
   Ry3 = np.array([[np.cos(theta3/2), -np.sin(theta3/2)],[np.sin(theta3/2), np.cos(theta3/2)]], dtype=complex)

   # Apply Ry to q_1
   q_1_rotated = Ry1 @ q_1

   # Construct the 2-qubit kron product q_1_rotated x q_0
   q_10 = np.kron(q_1_rotated, q_0)

   # Define the CNOT gate
   cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

   # Entangle q0 and q1
   q_10_entangled = cnot @ q_10

   # Apply Ry to q_2
   q_2_rotated = Ry2 @ q_2

   # tensor prouct q2 with entangled q1 and q0
   q_210 = np.kron(q_2_rotated, q_10_entangled)

   # tensor product cnot with identity
   cnot_3bits = np.kron(cnot, np.eye(2))

   # apply the second cnot entangling q_2 with entangled q_10
   q_210_entangled = cnot_3bits @ q_210

   # Finally, rotate q_3
   q_3_rotated = Ry3 @ q_3

   # tensor product q_3 with entangled q_210
   q_3210 = np.kron(q_3_rotated, q_210_entangled)

   # tensor product cnot with two identities
   cnot_4bits = np.kron(np.kron(cnot, np.eye(2)), np.eye(2))

   # Entangle q_3 with the rest
   q_3210_entangled = cnot_4bits @ q_3210

   return q_3210_entangled


# Calculate the expectation value <psi|H|psi>
def calculate_cost_function(statevector, Hamiltonian):
   return np.transpose(statevector) @ Hamiltonian @ statevector

# Create one function that can be called by the optimizer
def vqe_main(theta, Hamiltonian):
   # Create the statevector
   vec = generate_state_vector_advanced(theta)

   # Calculate value
   energy = calculate_cost_function(vec, Hamiltonian)

   # return
   return energy

def do_full_vqe(dist, starting_guess):
   # Specify the geometry in string format
   molkey = "H 0 0 0; H 0 0 "+str(dist)+" "

   # Feed this to pyscf and do an RHF calculation
   mol = gto.M(atom=molkey, basis="sto-3g")
   rhf_calc = scf.RHF(mol)
   rhf_calc.kernel()

   # Full CI
   # cisolver = fci.FCI(mol, rhf_calc.mo_coeff)
   # E_fci, _ = cisolver.kernel()
   # print("E_fci:", E_fci)

   # MOs and number of MOs
   mo_coeff = rhf_calc.mo_coeff
   num_orbitals = mo_coeff.shape[1]

   # Get the one electron and two electron integrals in MO basis
   h_core_ao = rhf_calc.get_hcore()
   h_core_mo = mo_coeff.T @ h_core_ao @ mo_coeff 
   #print(h_core_mo)

   eri_mo_packed = ao2mo.kernel(mol, mo_coeff)
   eri_mo = ao2mo.restore(1, eri_mo_packed, num_orbitals)

   # We have the spatial orbitals, but we need the spin component as well
   # So the total number of orbitals will double
   num_spin_orbitals = 2*num_orbitals
   h_spin = np.zeros((num_spin_orbitals, num_spin_orbitals))
   for p in range(num_orbitals):
      for q in range(num_orbitals):
         h_spin[2*p, 2*q] = h_core_mo[p,q] 
         h_spin[2*p+1,2*q+1] = h_core_mo[p,q]
   #print(h_spin)

   # Slightly more tricky to do this for the ERI
   # I'm gonna do it in the dumbest way possible
   eri_spin = np.zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals))
   for p in range(num_spin_orbitals):
      pw, p_spin_block  = np.divmod(p, 2)
      for q in range(num_spin_orbitals):
         qw, q_spin_block = np.divmod(q, 2)
         for r in range(num_spin_orbitals):
            rw, r_spin_block = np.divmod(r, 2)
            for s in range(num_spin_orbitals):
               sw, s_spin_block = np.divmod(s, 2)
               if (p_spin_block == r_spin_block and q_spin_block == s_spin_block):
                  eri_spin[p,q,r,s] = eri_mo[pw, qw, rw, sw]

   # We have the integrals. Now for the actually complicated part
   # The Jordan-Wigner transformation 


   # First we setup our 2x2 Pauli matrices
   I = np.eye(2)
   X = np.array([[0, 1.0], [1.0, 0.0]], dtype=complex)
   Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
   Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
   #print(Y)

   # Now we need to do tensor products of these 2x2 operators to produce operators that act on the joint 4-qubit Hilbert space
   X0_list = [I, I, I, X]
   X1_list = [I, I, X, I]
   X2_list = [I, X, I, I]
   X3_list = [X, I, I, I]

   Y0_list = [I, I, I, Y]
   Y1_list = [I, I, Y, I]
   Y2_list = [I, Y, I, I]
   Y3_list = [Y, I, I, I]

   Z0_list = [I, I, I, Z]
   Z1_list = [I, I, Z, I]
   Z2_list = [I, Z, I, I]
   Z3_list = [Z, I, I, I]

   X0 = ft.reduce(np.kron, X0_list)
   X1 = ft.reduce(np.kron, X1_list)
   X2 = ft.reduce(np.kron, X2_list)
   X3 = ft.reduce(np.kron, X3_list)
   X_operators = [X0, X1, X2, X3]

   Y0 = ft.reduce(np.kron, Y0_list)
   Y1 = ft.reduce(np.kron, Y1_list)
   Y2 = ft.reduce(np.kron, Y2_list)
   Y3 = ft.reduce(np.kron, Y3_list)
   Y_operators = [Y0, Y1, Y2, Y3]

   Z0 = ft.reduce(np.kron, Z0_list)
   Z1 = ft.reduce(np.kron, Z1_list)
   Z2 = ft.reduce(np.kron, Z2_list)
   Z3 = ft.reduce(np.kron, Z3_list)
   Z_operators = [Z0, Z1, Z2, Z3]

   # Now we will construct the creation and annihilation operators
   # Using Jordan-Wigner transform
   A = [] # annihilation
   Ap = []
   for i in range(num_spin_orbitals):
      ann_op = 0.5*(X_operators[i] + 1j*Y_operators[i])
      cre_op = 0.5*(X_operators[i] - 1j*Y_operators[i])
      for j in range(i):
         ann_op = Z_operators[j] @ ann_op
         cre_op = Z_operators[j] @ cre_op
      A.append(ann_op)
      Ap.append(cre_op)

   # Now we must put all the operators together to make the Hamiltonian
   # H = sum_ij (h_ij Ap_i A_j + sum_kl ( g_ijkl Ap_i Ap_j A_k A_l) )
   H_qubit = np.zeros((16, 16), dtype=complex)
   for i in range(num_spin_orbitals):
      for j in range(num_spin_orbitals):
         H_qubit = H_qubit + h_spin[i,j]* Ap[i] @ A[j]
         for k in range(num_spin_orbitals):
           for l in range(num_spin_orbitals):
             H_qubit = H_qubit + 0.5*eri_spin[i,j,k,l]* Ap[i] @ Ap[j] @ A[l] @ A[k]


   # Add nuclear energy
   E_nuclear = mol.energy_nuc()
   #print(E_nuclear)
   H_qubit = H_qubit + (E_nuclear * np.eye(16))

   #eigenvals, eigenvecs = np.linalg.eigh(H_qubit)
   #print(eigenvals)

   #psi_hf = np.zeros(16)
   #psi_hf[int("0011",2)] = 1

   #E_from_qubits = np.real(psi_hf @ H_qubit @ psi_hf)
   #print(E_from_qubits, rhf_calc.e_tot)
   # We now have the qubit Hamiltonian H_qubit, a 16x16 matrix. 
   # Now we have to implement the quantum gates and build the circuit

   #print("Qubit electronic energy:",
   #   np.real(psi_hf @ (H_qubit - E_nuclear*np.eye(16)) @ psi_hf))
   #print("PySCF electronic energy:", rhf_calc.e_tot - E_nuclear)


   # Now let's do a simple optimization
   #res = minimize_scalar(vqe_main, args=(H_qubit))
   #print(res.x, res.success, res.message)

   # Let's use the advanced ansatz
   start = starting_guess

   res = minimize(vqe_main, x0=start, args=(H_qubit))
   print(res)
   #print(vqe_main(np.pi, H_qubit))
   total_energy = vqe_main(res.x, H_qubit)
   return total_energy, res.x

def psi_theta(theta):
    c = np.cos(theta/2)
    s = np.sin(theta/2)
    # zero-based 16-vector, amplitude at index 0 (|0000>) and 3 (|0011>)
    psi = np.zeros(16, dtype=complex)
    psi[0] = c
    psi[3] = s
    return psi

def energy(theta, H):
    psi = psi_theta(theta)
    return np.real(np.vdot(psi, H @ psi))   # vdot conjugates first arg -> <psi| H |psi>

# Example usage:
# E = energy(np.pi, H_qubit)
# print("Validation=",E)

def param_shift_grad(theta, H):
    E_plus  = vqe_main(theta + np.pi/2, H)
    E_minus = vqe_main(theta - np.pi/2, H)
    return 0.5*(E_plus - E_minus)

# grad_at_pi = param_shift_grad(np.pi, H_qubit)
# print("dE/dθ at θ=π:", grad_at_pi)

def second_deriv_fd(theta, H, eps=1e-4):
    E_plus  = vqe_main(theta + eps, H)
    E_minus = vqe_main(theta - eps, H)
    E0      = vqe_main(theta, H)
    return (E_plus - 2*E0 + E_minus) / (eps**2)

# curv = second_deriv_fd(np.pi, H_qubit)
# print("Approx. second derivative at π:", curv)
# curv > 0 suggests local minimum, curv < 0 local maximum, curv ~ 0 saddle/flat

# thetas = np.linspace(np.pi-0.5, np.pi+0.5, 101)
# Es = [vqe_main(t,H_qubit) for t in thetas]
# min_idx = np.argmin(Es)
# print("Grid min at theta =", thetas[min_idx], "E =", Es[min_idx])


