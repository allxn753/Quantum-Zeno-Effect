import numpy as np
import qutip as qt


def single_qubit_resistor_hamiltonian(
    w_A: float,
    R: float,
    w_c: float,
    dw: float,
    C_g: float,
    C_A: float,
    lambda_A: float,
    n_modes: int,
    n_cut: int,
    hbar: float

) -> qt.Qobj:
    """
    Build the Hamiltonian for a transmon (qubit) coupled to a resistor bath model.

    Arguments:
        w_A: Transmon qubit angular frequency (rad/s).
        R: Resistor value (Ohms).
        w_c: Cutoff angular frequency (rad/s).
        dw: Frequency step (rad/s).
        C_g: Coupling capacitor capacitance (F).
        C_A: Qubit capacitance (F).
        lambda_A: A constant with the units of charge, depending on the features of 
            the transmon qubit, which allows us to pass from p_{A} to σ^{y}\_{A} 
            through p_{A} ≈ λ_{A} σ^{y}_{A} (units of charge).
        n_modes: Number of bath modes.
        n_cut: Fock truncation per mode (dimension of each oscillator Hilbert space).
        hbar: Reduced Planck constant (J*s).

    Returns:
        H: The full Hamiltonian as a QuTiP Qobj.
    """

    C_sum_A = C_A + C_g
    w_k = (np.arange(1, n_modes + 1)) * dw
    coef = 1j * lambda_A * (C_g / C_sum_A) * np.sqrt((R * hbar * w_k * (w_c**2) * dw) / (np.pi * (w_c**2 + w_k**2)))

    sz = qt.tensor([qt.sigmaz()], *([qt.qeye(n_cut)] * n_modes))
    sy = qt.tensor([qt.sigmay()], *([qt.qeye(n_cut)] * n_modes))

    a_ops = []
    a_dagger_a_ops = []
    for k in range(n_modes):
        identities = [qt.qeye(2)] + ([qt.qeye(n_cut)] * n_modes)
        identities[1 + k] = qt.destroy(n_cut)
        a_k_op = qt.tensor(identities)
        a_ops.append(a_k_op)
        a_dagger_a_ops.append(a_k_op.dag() * a_k_op)

    H = (0.5 * hbar * w_A * sz) + (sum(hbar * w_k[k] * a_dagger_a_ops[k] for k in range(n_modes))) + (sum(coef[k] * sy * (a_ops[k].dag() - a_ops[k]) for k in range(n_modes)))

    return H
