"""
Reproducing Figure 5 — Cattaneo et al., Ann. Phys. (Berlin) 2021, 533, 2100038

"""

import numpy as np
from scipy.integrate import quad
import qutip as qt
import matplotlib.pyplot as plt

# Parameters
omega1, omega2 = 1.0, 0.99
mu      = 10**(-1.5)
beta    = 10.0
T1_inv  = 1.0 / 3e5
omega_C = 20.0

# Ohmic spectral density & Bose–Einstein factor
def J(w):   return mu**2 * w * omega_C**2 / (omega_C**2 + w**2)
def N_B(w):
    a = beta * w
    return 0.0 if a > 700 else (1/a if a < 1e-12 else 1/(np.exp(a) - 1))

# Fourier transforms
def Gamma_pos(oq):
    re = np.pi * (N_B(oq) + 1) * J(oq)
    f  = lambda w: J(w) * ((N_B(w)+1)/(oq-w+1e-30) + N_B(w)/(oq+w))
    v1,_ = quad(f, 1e-7, oq-1e-4, limit=600, epsabs=1e-13)
    v2,_ = quad(f, oq+1e-4, 1e4,  limit=600, epsabs=1e-13)
    return re + 1j*(v1+v2)

def Gamma_neg(oq):
    re    = np.pi * N_B(oq) * J(oq)
    f_pv  = lambda w: J(w) * N_B(w) / (oq-w+1e-30)
    f_reg = lambda w: -J(w) * (N_B(w)+1) / (oq+w)
    v1,_ = quad(f_pv,  1e-7, oq-1e-4, limit=600, epsabs=1e-13)
    v2,_ = quad(f_pv,  oq+1e-4, 1e4,  limit=600, epsabs=1e-13)
    vr,_ = quad(f_reg, 1e-7, 1e4,     limit=600, epsabs=1e-13)
    return re + 1j*(v1+v2+vr)

G  = [Gamma_pos(omega1), Gamma_pos(omega2)]
Gn = [Gamma_neg(omega1), Gamma_neg(omega2)]

# Master-equation coefficients
def gd(j,k):
    v = G[j] + np.conj(G[k])
    if j == k: v += T1_inv
    return complex(v)
def gu(j,k): return complex(Gn[j] + np.conj(Gn[k]))
def sd(j,k): return complex((G[j]  - np.conj(G[k]))  / (2j))
def su(j,k): return complex((Gn[j] - np.conj(Gn[k])) / (2j))

# Operators
I = qt.qeye(2)
sp1, sm1 = qt.tensor(qt.sigmap(), I), qt.tensor(qt.sigmam(), I)
sp2, sm2 = qt.tensor(I, qt.sigmap()), qt.tensor(I, qt.sigmam())
sx1, sx2 = qt.tensor(qt.sigmax(), I), qt.tensor(I, qt.sigmax())
Pe1 = qt.tensor(qt.basis(2,0)*qt.basis(2,0).dag(), I)
Pe2 = qt.tensor(I, qt.basis(2,0)*qt.basis(2,0).dag())
spl, sml = [sp1,sp2], [sm1,sm2]

# System Hamiltonian
H = ( 0.5*(omega1*qt.tensor(qt.sigmaz(),I) + omega2*qt.tensor(I,qt.sigmaz()))
    + sum(sd(j,k)*spl[k]*sml[j] + su(j,k)*sml[k]*spl[j]
          for j in range(2) for k in range(2)) )

# Jump operators
def jump_ops(rate_fn, base_ops):
    G_mat = np.array([[rate_fn(j,k) for k in range(2)] for j in range(2)])
    G_mat = 0.5*(G_mat + G_mat.conj().T)
    try:    L = np.linalg.cholesky(G_mat + 1e-14*np.eye(2))
    except: evals,evecs = np.linalg.eigh(G_mat); L = evecs@np.diag(np.maximum(evals,0)**.5)
    return [L[0,m]*base_ops[0] + L[1,m]*base_ops[1]
            for m in range(2) if (L[0,m]*base_ops[0]+L[1,m]*base_ops[1]).norm() > 1e-14]

c_ops = jump_ops(gd, sml) + jump_ops(gu, spl)

rho_Syn = qt.ket2dm(qt.tensor(
    np.cos(np.pi/4)*qt.basis(2,0) + np.sin(np.pi/4)*qt.basis(2,1),
    np.cos(np.pi/3)*qt.basis(2,0) + 1j*np.sin(np.pi/3)*qt.basis(2,1)))

rho_sub = qt.ket2dm(
    (qt.tensor(qt.basis(2,0),qt.basis(2,1))
   - qt.tensor(qt.basis(2,1),qt.basis(2,0))) / np.sqrt(2))

psi_p   = (qt.basis(2,0) + qt.basis(2,1)) / np.sqrt(2)
rho_C   = qt.ket2dm(qt.tensor(psi_p, psi_p))

# Figure 5a data
dt    = 0.1
t_fin = np.arange(0.0, 1001.0 + dt, dt)
res_a = qt.mesolve(H, rho_Syn, t_fin, c_ops, e_ops=[sx1, sx2])
s1, s2 = np.array(res_a.expect[0]), np.array(res_a.expect[1])

win = int(round(7.0 / dt))   # Δt = 7/ω₁ window
def pearson(i):
    a, b = s1[i:i+win]-s1[i:i+win].mean(), s2[i:i+win]-s2[i:i+win].mean()
    d = np.sqrt((a**2).sum()*(b**2).sum())
    return float((a*b).sum()/d) if d > 1e-20 else 0.0

stride = 10   # evaluate every 1/ω₁
t_eval = t_fin[::stride]
C_arr  = np.array([pearson(i*stride) for i in range(len(t_eval))])

# Figure 5b data
t_sub = np.linspace(0, 1000, 500)
res_b = qt.mesolve(H, rho_sub, t_sub, c_ops, e_ops=[Pe1, Pe2])
Pe1_t = np.array(res_b.expect[0])
Pe2_t = np.array(res_b.expect[1])

# Figure 5c data
t_neg = np.linspace(0, 100, 150)
res_c = qt.mesolve(H, rho_C, t_neg, c_ops)
neg_t = np.array([qt.negativity(r, 1) for r in res_c.states])
N_M, t_NM = neg_t.max(), t_neg[neg_t.argmax()]
print(f"  N_M = {N_M:.4f} at t = {t_NM:.1f}  (paper: 0.37 at t=21)")

def collectiveness(t):
    U = qt.propagator(H, t, c_ops).full()
    n = 4
    Phi = np.zeros((n**2, n**2), complex)
    for IS in range(n):
        for JS in range(n):
            for ISp in range(n):
                for JSp in range(n):
                    Phi[IS*n+ISp, JS*n+JSp] = 0.25 * U[IS*n+JS, ISp*n+JSp]
    Phi_q = qt.Qobj(Phi, dims=[[2,2,2,2],[2,2,2,2]])
    rho_11p = qt.ptrace(Phi_q, [0, 2])
    rho_22p = qt.ptrace(Phi_q, [1, 3])
    MI = (qt.entropy_vn(rho_11p) + qt.entropy_vn(rho_22p)
        - qt.entropy_vn(Phi_q))
    return MI / (4 * np.log(2))

t_coll = np.unique(np.concatenate([np.linspace(0,20,10),
                                    np.linspace(20,60,10),
                                    np.linspace(60,100,6)]))
C_coll = np.array([collectiveness(t) for t in t_coll])
I_M, t_IM = C_coll.max(), t_coll[C_coll.argmax()]
print(f"  I_M = {I_M:.4f} at t = {t_IM:.1f}  (paper: 0.81)")

# Plots
BLUE, ORANGE = '#1f77b4', '#ff7f0e'
RED,  CYAN   = '#d62728', '#17becf'
PURPLE       = '#9467bd'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.linewidth': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})

# Fig 5a plot
fig_a, ax_a = plt.subplots(figsize=(7, 4.5))
ax_a.plot(t_eval, C_arr, color='blue')
ax_a.axhline(0,  color='gray', lw=0.5, ls='--', alpha=0.6)
ax_a.axhline(-1, color='gray', lw=0.4, ls=':',  alpha=0.4)
ax_a.set(xlim=(0,1000), ylim=(-1.1,0.65),
         xlabel=r'$t\omega_1$', ylabel=r'$\mathcal{C}_{\Delta t}(t)$')
ax_a.set_title('Pearson Coefficient', loc='left', fontweight='bold')
fig_a.tight_layout()
fig_a.savefig('fig5a_pearson.png')
print("Saved: fig5a_pearson.png")

# Fig 5a subplot 1
fig_a2, ax_a2 = plt.subplots(figsize=(5, 3.5))
m1 = (t_fin >= 800) & (t_fin <= 850)
ax_a2.plot(t_fin[m1], s1[m1], color=BLUE,   lw=1.5, label=r'$\langle\sigma_1^x(t)\rangle$')
ax_a2.plot(t_fin[m1], s2[m1], color=ORANGE, lw=1.5, label=r'$\langle\sigma_2^x(t)\rangle$')
ax_a2.set(xlim=(800,850), xlabel=r'$t\omega_1$', ylabel=r'$\langle\sigma^x\rangle$')
ax_a2.set_title("Time evolution of sigma_1 and sigma_2", loc='left', fontweight='bold')
ax_a2.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='lightgray')
fig_a2.tight_layout()
fig_a2.savefig('fig5a_sync_late.png')
print("Saved: fig5a_sync_late.png")

# Fig 5a subplot 2
fig_a3, ax_a3 = plt.subplots(figsize=(5, 3.5))
m2 = (t_fin >= 100) & (t_fin <= 150)
ax_a3.plot(t_fin[m2], s1[m2], color=BLUE,   lw=1.5, label=r'$\langle\sigma_1^x(t)\rangle$')
ax_a3.plot(t_fin[m2], s2[m2], color=ORANGE, lw=1.5, label=r'$\langle\sigma_2^x(t)\rangle$')
ax_a3.set(xlim=(100,150), xlabel=r'$t\omega_1$', ylabel=r'$\langle\sigma^x\rangle$')
ax_a3.set_title("Time evolution of sigma_1 and sigma_2", loc='left', fontweight='bold')
ax_a3.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='lightgray')
fig_a3.tight_layout()
fig_a3.savefig('fig5a_sync_early.png')
print("Saved: fig5a_sync_early.png")

# Fig 5b plot
fig_b, ax_b = plt.subplots(figsize=(6, 4.5))
ax_b.plot(t_sub, Pe1_t, color=RED,  lw=1.5, label=r'$\langle P_1^e(t)\rangle$')
ax_b.plot(t_sub, Pe2_t, color=BLUE, lw=1.5, label=r'$\langle P_2^e(t)\rangle$')
ax_b.set(xlim=(0,1000), ylim=(0.40,0.62),
         xlabel=r'$t\omega_1$', ylabel='Population')
ax_b.set_title('Time Evolution of the Excited State Populations', loc='left', fontweight='bold')
ax_b.legend(fontsize=10, loc='upper right', framealpha=0.9, edgecolor='lightgray')
fig_b.tight_layout()
fig_b.savefig('fig5b_subradiance.png')
print("Saved: fig5b_subradiance.png")

# Fig 5c plot
fig_c, ax_c = plt.subplots(figsize=(6, 4.5))
ax_c.plot(t_neg,  neg_t,  color=CYAN,   lw=2.0, label=r'$\mathcal{N}(t)$')
ax_c.plot(t_coll, C_coll, color=PURPLE, lw=2.0, label=r'$\bar{I}(\mathcal{E}(t))$')
ax_c.set(xlim=(0,100), ylim=(-0.02,1.0),
         xlabel=r'$t\omega_1$', ylabel='Value')
ax_c.set_title('Evolution of Negativity and Collectiveness', loc='left', fontweight='bold')
ax_c.legend(fontsize=10, loc='upper right', framealpha=0.9, edgecolor='lightgray')
fig_c.tight_layout()
fig_c.savefig('fig5c_entanglement.png')
print("Saved: fig5c_entanglement.png")