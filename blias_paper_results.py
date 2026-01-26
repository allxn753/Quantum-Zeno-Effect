import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# ====================
# Problem Parameters (h-bar = 1)
# ====================
N = 15  # number of cavity fock states
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.3 * 2 * np.pi  # coupling strength
kappa = 0.005  # cavity dissipation rate
gamma = 0.05  # atom dissipation rate
n_th_a = 0.0  # temperature in frequency units
use_rwa = False

def gamma_phi(omega):
    return 0.005

tlist = np.linspace(0, 40, 200)

a  = qt.tensor(qt.destroy(N), qt.qeye(2))
sm = qt.tensor(qt.qeye(N), qt.destroy(2))
sz = qt.tensor(qt.qeye(N), qt.sigmaz())

psi0 = qt.tensor(qt.basis(N, 0), qt.basis(2, 0))

H = (wc * a.dag() * a + wa / 2 * sz + g * (a + a.dag()) * (sm + sm.dag())) # Rabi Hamiltonian

# ====================
# SME
# ====================
c_ops_std = []
c_ops_std.append(np.sqrt(kappa) * a)
c_ops_std.append(np.sqrt(gamma) * sm)

out_std = qt.mesolve(H, psi0, tlist, c_ops_std, [a.dag() * a, sm.dag() * sm])

# ====================
# DME
# ====================
evals, estates = H.eigenstates()

c_ops_dressed = []

for i in range(len(evals)):

    phi_i = np.sqrt(gamma_phi(0)/2) * estates[i].dag() * sm * estates[i]
    c_ops_dressed.append(phi_i * (estates[i] * estates[i].dag()))
    
    for j in range(len(evals)):

        delta_ij = abs(evals[i] - evals[j])

        if evals[j] > evals[i]:

            cav_ij  = estates[i].dag() * (a + a.dag()) * estates[j]
            atom_ij = estates[i].dag() * (sm + sm.dag()) * estates[j]

            c_ops_dressed.append(kappa * abs(cav_ij)**2 * (estates[i] * estates[j].dag()))
            c_ops_dressed.append(gamma * abs(atom_ij)**2 * (estates[i] * estates[j].dag()))
        
        if evals[j] != evals[i]:

            atom_ij = estates[i].dag() * sm * estates[j]

            c_ops_dressed.append(gamma_phi(delta_ij/2) * abs(atom_ij)**2 * (estates[i] * estates[j].dag()))

out_dressed = qt.mesolve(H, psi0, tlist, c_ops_dressed, [a.dag() * a, sm.dag() * sm])

# ====================
# Plotting Everything
# ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].plot(tlist, out_std.expect[0], "b", label=r"$\langle a^\dagger a \rangle$")
axes[0].plot(tlist, out_std.expect[1], "r", label=r"$\langle \sigma_+\sigma_- \rangle$")
axes[0].set_title("Standard Master Equation")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Occupation Probability")
axes[0].legend()

axes[1].plot(tlist, out_dressed.expect[0], "b", label=r"$\langle a^\dagger a \rangle$")
axes[1].plot(tlist, out_dressed.expect[1], "r", label=r"$\langle \sigma_+\sigma_- \rangle$")
axes[1].set_title("Dressed Master Equation")
axes[1].set_xlabel("Time")
axes[1].legend()

plt.tight_layout()
plt.show()
