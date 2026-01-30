import numpy as np
import qutip as qt

# ====================
# Problem Parameters
# ====================


def single_qubit_resistor_hamiltonian(
    w_q,               # qubit frequency ωA (in your units; QuTiP usually sets ħ=1)
    R,                 # resistance R
    w_c,               # cutoff frequency ωC
    C_g,               # coupling capacitance Cg
    C_A,               # qubit capacitance CA
    lambda_A,      # λA mapping pA ≈ λA σ_y (paper-specific constant)
    n_modes,        # number of bath modes to keep (finite approximation)
    n_cut,           # Fock truncation for each bath oscillator
    w_max,        # max bath freq; if None, use ~ several cutoffs
    hbar,          # keep explicit if you want SI-like scaling; default ħ=1
):

    C_sum_A = C_A + C_g

    dw = w_max / n_modes
    w_k = dw * (np.arange(1, n_modes + 1))

    coef = 1j * lambda_A * (C_g / C_sum_A) * np.sqrt((R * hbar * w_k * (w_c**2) * dw) / (np.pi * (w_c**2 + w_k**2)))

    sz = qt.tensor([qt.sigmaz()], *([qt.qeye(n_cut)] * n_modes))
    sy = qt.tensor([qt.sigmay()], *([qt.qeye(n_cut)] * n_modes))

    # Mode annihilators (each embedded into the full tensor space)
    a_ops = []
    n_ops = []
    for k in range(n_modes):
        factors = [qt.qeye(2)] + [qt.qeye(n_cut)] * n_modes
        factors[1 + k] = qt.destroy(n_cut)
        a_k_op = qt.tensor(factors)
        a_ops.append(a_k_op)
        n_ops.append(a_k_op.dag() * a_k_op)

    # ====================
    # Hamiltonians
    # ====================
    H_qubit = 0.5 * hbar * w_q * sz
    H_resistor  = sum(hbar * w_k[k] * n_ops[k] for k in range(n_modes))
    H_interaction = sum(coef[k] * sy * (a_ops[k].dag() - a_ops[k]) for k in range(n_modes))

    H = H_qubit + H_resistor + H_interaction

    return H
