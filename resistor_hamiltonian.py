import numpy as np
import qutip as qt

N = 15  # number of cavity fock states
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.3 * 2 * np.pi  # coupling strength
kappa = 0.005  # cavity dissipation rate
gamma = 0.05  # atom dissipation rate
n_th_a = 0.0  # temperature in frequency units
use_rwa = False

a  = qt.tensor(qt.destroy(N), qt.qeye(2))
sm = qt.tensor(qt.qeye(N), qt.destroy(2))
sz = qt.tensor(qt.qeye(N), qt.sigmaz())

def single_qubit_resistor_hamiltonian():

    C_sum_A = C_A + C_g

    H = (p_A**2)/(2 * C_sum_A) + V(phi_A)

    H = ((hbar * omega_A)/2) * sz_A

    return