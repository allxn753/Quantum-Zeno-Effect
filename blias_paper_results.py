import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

'''
# System parameters
wr = 2 * np.pi * 6.0e9     # resonator frequency
wq = 2 * np.pi * 6.0e9     # qubit frequency
kappa = 2 * np.pi * 0.1e6 # cavity decay
gamma = 2 * np.pi * 0.1e6 # qubit relaxation
N = 10                    # cavity truncation

a = qt.destroy(N)
sx = qt.sigmax()
sz = qt.sigmaz()
sm = qt.sigmam()
I_c = qt.qeye(N)
I_q = qt.qeye(2)

def rabi_hamiltonian(g):
    return (
        wr * qt.tensor(a.dag()*a, I_q)
        + 0.5 * wq * qt.tensor(I_c, sz)
        + g * qt.tensor(a + a.dag(), sx)
    )

def standard_liouvillian(g):
    H = rabi_hamiltonian(g)
    c_ops = [
        np.sqrt(kappa) * qt.tensor(a, I_q),
        np.sqrt(gamma) * qt.tensor(I_c, sm)
    ]
    return H, c_ops

def dressed_liouvillian(g, eps=1e-6):
    H = rabi_hamiltonian(g)
    evals, evecs = H.eigenstates()

    c_ops = []

    X_op = qt.tensor(a + a.dag(), I_q)
    sx_op = qt.tensor(I_c, sx)

    for j in range(len(evals)):
        for k in range(j):
            wjk = evals[j] - evals[k]
            if wjk <= 0:
                continue

            # cavity-induced decay
            xjk = abs(evecs[j].dag() * X_op * evecs[k])**2
            if xjk > 1e-12:
                c_ops.append(np.sqrt(kappa * xjk) * evecs[k] * evecs[j].dag())

            # qubit-induced decay
            sjk = abs(evecs[j].dag() * sx_op * evecs[k])**2
            if sjk > 1e-12:
                c_ops.append(np.sqrt(gamma * sjk) * evecs[k] * evecs[j].dag())

    # ---- CRUCIAL REGULARIZATION ----
    # infinitesimal global loss
    c_ops.append(np.sqrt(eps) * qt.tensor(a, I_q))

    return H, c_ops


g_vals = np.linspace(0, 0.4 * wr, 15)
n_std = []
n_dressed = []

for g in g_vals:
    Hs, cs = standard_liouvillian(g)
    Hd, cd = dressed_liouvillian(g)

    rho0 = qt.tensor(qt.basis(N,0), qt.basis(2,0)).proj()

    rho_ss_std = qt.steadystate(Hs, cs)
    rho_ss_dr  = qt.steadystate(Hd, cd)

    n_op = qt.tensor(a.dag()*a, I_q)
    n_std.append(qt.expect(n_op, rho_ss_std))
    n_dressed.append(qt.expect(n_op, rho_ss_dr))

plt.plot(g_vals / wr, n_std, label="Standard ME")
plt.plot(g_vals / wr, n_dressed, "--", label="Dressed ME")
plt.xlabel(r"g / $\omega$")
plt.ylabel(r"$\langle a^\dagger a \rangle$")
plt.legend()
plt.show()
'''

N = 15  # number of cavity fock states
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.05 * 2 * np.pi  # coupling strength
kappa = 0.005  # cavity dissipation rate
gamma = 0.05  # atom dissipation rate
n_th_a = 0.0  # temperature in frequency units
use_rwa = True

tlist = np.linspace(0, 40, 100)

# intial state
psi0 = qt.tensor(qt.basis(N, 0), qt.basis(2, 0))

# collapse operators
a = qt.tensor(qt.destroy(N), qt.qeye(2))
sm = qt.tensor(qt.qeye(N), qt.destroy(2).dag())
sz = qt.tensor(qt.qeye(N), qt.sigmaz())

# Hamiltonian
if use_rwa:
    H = wc * a.dag() * a + wa / 2 * sz + g * (a.dag() * sm + a * sm.dag())
else:
    H = wc * a.dag() * a + wa / 2 * sz + g * (a.dag() + a) * (sm + sm.dag())

c_op_list = []

# Photon annihilation
rate = kappa * (1 + n_th_a)
c_op_list.append(np.sqrt(rate) * a)

# Photon creation
rate = kappa * n_th_a
c_op_list.append(np.sqrt(rate) * a.dag())

# Atom annihilation
rate = gamma
c_op_list.append(np.sqrt(rate) * sm)

output = qt.mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tlist, output.expect[0], label="Cavity")
ax.plot(tlist, output.expect[1], label="Atom excited state")
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Occupation probability")
ax.set_title("Vacuum Rabi oscillations at T={}".format(n_th_a))
fig.show()