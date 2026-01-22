import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

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

def dressed_liouvillian(g):
    H = rabi_hamiltonian(g)
    evals, evecs = H.eigenstates()

    c_ops = []
    for j in range(len(evals)):
        for k in range(j):
            wjk = evals[j] - evals[k]
            if wjk > 0:
                op = qt.tensor(a + a.dag(), I_q)
                matel = abs(evecs[j].dag() * op * evecs[k])**2
                rate = kappa * matel
                if rate > 1e-12:
                    c_ops.append(np.sqrt(rate) * evecs[k] * evecs[j].dag())
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
plt.xlabel("g / ω")
plt.ylabel("⟨a†a⟩")
plt.legend()
plt.show()
