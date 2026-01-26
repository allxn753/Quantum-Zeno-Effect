import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# ====================
# Parameters
# ====================
N = 15  # number of cavity fock states
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.3 * 2 * np.pi  # coupling strength
kappa = 0.005  # cavity dissipation rate
gamma = 0.05  # atom dissipation rate
n_th_a = 0.0  # temperature in frequency units
use_rwa = False

tlist = np.linspace(0, 40, 200)


# ====================
# Operators
# ====================
a  = qt.tensor(qt.destroy(N), qt.qeye(2))
sm = qt.tensor(qt.qeye(N), qt.destroy(2))
sz = qt.tensor(qt.qeye(N), qt.sigmaz())

# ====================
# Initial state
# ====================
psi0 = qt.tensor(qt.basis(N, 0), qt.basis(2, 0))

# ====================
# Full Rabi Hamiltonian
# ====================
H = (wc * a.dag() * a + wa / 2 * sz + g * (a + a.dag()) * (sm + sm.dag()))

# ====================
# STANDARD master equation
# ====================
c_ops_std = []
c_ops_std.append(np.sqrt(kappa) * a)
c_ops_std.append(np.sqrt(gamma) * sm)

out_std = qt.mesolve(H, psi0, tlist, c_ops_std, [a.dag() * a, sm.dag() * sm])

# ====================
# DRESSED master equation
# ====================
evals, evecs = H.eigenstates()

X_cav  = a + a.dag()
X_atom = sm + sm.dag()

c_ops_dressed = []

for i in range(len(evals)):
    for j in range(len(evals)):
        if evals[j] > evals[i]:
            # Transition frequency
            w_ij = evals[j] - evals[i]

            cav_ij  = evecs[i].dag() * X_cav  * evecs[j]
            atom_ij = evecs[i].dag() * X_atom * evecs[j]


            if abs(cav_ij) > 1e-6:
                c_ops_dressed.append(
                    np.sqrt(kappa * abs(cav_ij)**2)
                    * (evecs[i] * evecs[j].dag())
                )

            if abs(atom_ij) > 1e-6:
                c_ops_dressed.append(
                    np.sqrt(gamma * abs(atom_ij)**2)
                    * (evecs[i] * evecs[j].dag())
                )

out_dressed = qt.mesolve(H, psi0, tlist, c_ops_dressed, [a.dag() * a, sm.dag() * sm])

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# ====================
# Standard ME plot
# ====================
axes[0].plot(tlist, out_std.expect[0], "b--", label=r"$\langle a^\dagger a \rangle$")
axes[0].plot(tlist, out_std.expect[1], "r--", label=r"$\langle \sigma_+\sigma_- \rangle$")
axes[0].set_title("Standard master equation (bare operators)")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Occupation")
axes[0].legend()

# ====================
# Dressed ME plot
# ====================
axes[1].plot(tlist, out_dressed.expect[0], "b", label=r"$\langle a^\dagger a \rangle$")
axes[1].plot(tlist, out_dressed.expect[1], "r", label=r"$\langle \sigma_+\sigma_- \rangle$")
axes[1].set_title("Dressed master equation (eigenbasis)")
axes[1].set_xlabel("Time")
axes[1].legend()

plt.tight_layout()
plt.show()
