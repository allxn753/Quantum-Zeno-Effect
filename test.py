"""
Figure 5 — Cattaneo et al., Ann. Phys. (Berlin) 2021, 533, 2100038
Two transmon qubits coupled to a common resistor (Ohmic bath).

Libraries used
--------------
qutip      — quantum operators, Liouvillian, time evolution (mesolve),
             expectation values, negativity
matplotlib — all plotting
scipy.integrate.quad — the only SciPy function needed, for the two
             principal-value integrals that define Im[Γ(±ω)].
             QuTiP has no built-in function for computing bath spectral
             integrals; this is always done analytically or numerically
             by the user before passing rates into the Lindblad equation.
numpy      — QuTiP requires numpy internally; we use it here only for
             lightweight array ops (np.exp, np.linspace, np.cos/sin)
             that QuTiP does not expose as standalone utilities.

Outputs
-------
fig5a_pearson.png
fig5b_subradiance.png
fig5c_entanglement.png
"""

import numpy as np
from scipy.integrate import quad      # for principal-value bath integrals only
import qutip as qt
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Physical parameters  (ℏ = 1, ω₁ = 1)
# ═══════════════════════════════════════════════════════════════════════════════
omega1  = 1.0          # qubit 1 frequency (sets the unit)
omega2  = 0.99         # qubit 2 frequency  (Δω = 0.01 ω₁)
mu      = 10**(-1.5)   # system–bath coupling constant  (~0.032)
beta    = 10.0         # inverse temperature  (β ℏ ω₁ = 10  →  T ≈ 24 mK)
T1_inv  = 1.0 / 3e5   # local (phenomenological) decay rate  1/T₁
omega_C = 20.0         # Ohmic spectral-density cut-off  ω_C = 20 ω₁

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Bath spectral density and Bose–Einstein factor
# ═══════════════════════════════════════════════════════════════════════════════
def J(w):
    """Ohmic spectral density  J(ω)  (Eq. A4)."""
    return mu**2 * w * omega_C**2 / (omega_C**2 + w**2)

def N_B(w):
    """Bose–Einstein occupation number."""
    a = beta * w
    if a > 700:    return 0.0
    if a < 1e-12:  return 1.0 / a
    return 1.0 / (np.exp(a) - 1.0)

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  One-sided Fourier transform  Γ_β(±ω)  (Eq. A2)
#
#     Γ_β(+ω) = π(N(ω)+1)J(ω)  +  i · PV∫ J(ω')[(N+1)/(ω-ω') + N/(ω+ω')] dω'
#
#     Γ_β(−ω) = π N(ω) J(ω)    +  i · PV∫ J(ω')[N/(ω-ω') − (N+1)/(ω+ω')] dω'
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#               NOTE: the imaginary part of Γ(−ω) is NOT simply −Im[Γ(+ω)].
#               Both integrals are computed via scipy.quad (Cauchy PV).
# ═══════════════════════════════════════════════════════════════════════════════
def Gamma_pos(oq):
    """Γ_β(+ω_q) for ω_q > 0."""
    re  = np.pi * (N_B(oq) + 1) * J(oq)
    f   = lambda wp: J(wp) * ((N_B(wp)+1)/(oq-wp+1e-30) + N_B(wp)/(oq+wp))
    v1, _ = quad(f, 1e-7, oq - 1e-4, limit=600, epsabs=1e-13)
    v2, _ = quad(f, oq + 1e-4, 1e4,  limit=600, epsabs=1e-13)
    return re + 1j*(v1 + v2)

def Gamma_neg(oq):
    """Γ_β(−ω_q) for ω_q > 0  [corrected imaginary part]."""
    re     = np.pi * N_B(oq) * J(oq)
    f_pv   = lambda wp: J(wp) * N_B(wp) / (oq - wp + 1e-30)   # Cauchy singular
    f_reg  = lambda wp: -J(wp) * (N_B(wp)+1) / (oq + wp)       # regular
    eps    = 1e-4
    v1, _ = quad(f_pv,  1e-7, oq - eps, limit=600, epsabs=1e-13)
    v2, _ = quad(f_pv,  oq + eps, 1e4,  limit=600, epsabs=1e-13)
    vr, _ = quad(f_reg, 1e-7, 1e4,      limit=600, epsabs=1e-13)
    return re + 1j*(v1 + v2 + vr)

print("Computing bath integrals Gamma(±omega1), Gamma(±omega2)…")
G  = [Gamma_pos(omega1), Gamma_pos(omega2)]   # Γ(+ω_j)
Gn = [Gamma_neg(omega1), Gamma_neg(omega2)]   # Γ(−ω_j)

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Master-equation coefficients  (Eq. A1)
#
#     γ↓_{jk} = g_j g_k [Γ(ω_j) + Γ*(ω_k)] + δ_{jk}/T₁
#     γ↑_{jk} = g_j g_k [Γ(−ω_j) + Γ*(−ω_k)]
#     s↓_{jk} = g_j g_k [Γ(ω_j) − Γ*(ω_k)] / 2i
#     s↑_{jk} = g_j g_k [Γ(−ω_j) − Γ*(−ω_k)] / 2i
# ═══════════════════════════════════════════════════════════════════════════════
def gd(j, k):
    v = G[j] + np.conj(G[k])
    if j == k: v += T1_inv
    return complex(v)

def gu(j, k): return complex(Gn[j] + np.conj(Gn[k]))
def sd(j, k): return complex((G[j]  - np.conj(G[k]))  / (2j))
def su(j, k): return complex((Gn[j] - np.conj(Gn[k])) / (2j))

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  QuTiP operators
#
#     Basis: |e⟩ = qt.basis(2,0),  |g⟩ = qt.basis(2,1)
#     Two-qubit order: qubit 1 ⊗ qubit 2
#     States: |ee⟩=0, |eg⟩=1, |ge⟩=2, |gg⟩=3  (QuTiP tensor ordering)
# ═══════════════════════════════════════════════════════════════════════════════
I   = qt.qeye(2)
sz  = qt.sigmaz()          #  σ_z  = diag(+1, −1)
sp  = qt.sigmap()          #  σ₊   = |e⟩⟨g|
sm  = qt.sigmam()          #  σ₋   = |g⟩⟨e|
sx  = qt.sigmax()          #  σ_x  = σ₊ + σ₋

# Embedded in the two-qubit space
sz1 = qt.tensor(sz, I);   sz2 = qt.tensor(I, sz)
sp1 = qt.tensor(sp, I);   sp2 = qt.tensor(I, sp)
sm1 = qt.tensor(sm, I);   sm2 = qt.tensor(I, sm)
sx1 = qt.tensor(sx, I);   sx2 = qt.tensor(I, sx)

spl = [sp1, sp2]           # list for easy indexing
sml = [sm1, sm2]

# Excited-state projectors  P^e_k = |e⟩⟨e| on qubit k
Pe1 = qt.tensor(qt.basis(2,0) * qt.basis(2,0).dag(), I)
Pe2 = qt.tensor(I, qt.basis(2,0) * qt.basis(2,0).dag())

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Hamiltonian  H = H_S + H_LS  (Eqs. 1–3)
# ═══════════════════════════════════════════════════════════════════════════════
H_S  = 0.5 * (omega1 * sz1 + omega2 * sz2)

H_LS = sum(
    sd(j,k) * spl[k] * sml[j]    # s↓_{jk} σ⁺_k σ⁻_j
  + su(j,k) * sml[k] * spl[j]    # s↑_{jk} σ⁻_k σ⁺_j
    for j in range(2) for k in range(2)
)

H = H_S + H_LS

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Collective jump operators  (Eq. 4)
#
#     The dissipator has correlated off-diagonal rates γ_{jk} with j ≠ k.
#     qt.liouvillian() expects individual jump operators, so we Cholesky-
#     decompose the 2×2 rate matrices γ↓ and γ↑ into collective operators:
#
#         C_m = Σ_j  L_{jm} · base_op_j        (L = Cholesky factor)
#
#     Then  Σ_{jk} γ_{jk} D[base_j, base_k]  =  Σ_m D[C_m]  exactly.
# ═══════════════════════════════════════════════════════════════════════════════
def cholesky_jump_ops(rate_fn, base_ops):
    """Return collective jump operators from a 2×2 rate matrix via Cholesky."""
    G_mat = np.array([[rate_fn(j,k) for k in range(2)] for j in range(2)])
    G_mat = 0.5 * (G_mat + G_mat.conj().T)          # enforce Hermitian
    try:
        L = np.linalg.cholesky(G_mat + 1e-14*np.eye(2))
    except np.linalg.LinAlgError:
        ev, vc = np.linalg.eigh(G_mat)
        L = vc @ np.diag(np.maximum(ev, 0)**0.5)    # fallback: matrix sqrt
    ops = []
    for m in range(2):
        C = L[0,m] * base_ops[0] + L[1,m] * base_ops[1]
        if C.norm() > 1e-14:
            ops.append(C)
    return ops

c_ops = cholesky_jump_ops(gd, sml) + cholesky_jump_ops(gu, spl)
print(f"Jump operators: {len(c_ops)} total  ({len(cholesky_jump_ops(gd,sml))} down, "
      f"{len(cholesky_jump_ops(gu,spl))} up)")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Initial states
# ═══════════════════════════════════════════════════════════════════════════════
# (a) Synchronization — separable, large coherences, qubit-asymmetric (Sec. 3.1)
psi_q1  = np.cos(np.pi/4)*qt.basis(2,0) + np.sin(np.pi/4)*qt.basis(2,1)
psi_q2  = np.cos(np.pi/3)*qt.basis(2,0) + 1j*np.sin(np.pi/3)*qt.basis(2,1)
rho_Syn = qt.ket2dm(qt.tensor(psi_q1, psi_q2))

# (b) Subradiance — Bell singlet  (|eg⟩ − |ge⟩)/√2  (Sec. 3.2)
psi_sub = (qt.tensor(qt.basis(2,0), qt.basis(2,1))
         - qt.tensor(qt.basis(2,1), qt.basis(2,0))) / np.sqrt(2)
rho_sub = qt.ket2dm(psi_sub)

# (c) Entanglement — maximally coherent state  |+⟩⟨+|  (Sec. 3.3)
psi_plus = (qt.basis(2,0) + qt.basis(2,1)) / np.sqrt(2)
rho_C    = qt.ket2dm(qt.tensor(psi_plus, psi_plus))

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Panel (a) — Pearson correlation coefficient  C_{Δt}(t)
#
#     Fine grid (dt = 0.1/ω₁) to resolve oscillations within the window.
#     Sliding window: Δt = 7/ω₁  (from paper Fig. 5 caption).
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel (a): evolving <sigma^x> on fine grid…")
dt_fine = 0.1
t_fine  = np.arange(0.0, 1001.0 + dt_fine, dt_fine)

result_a = qt.mesolve(H, rho_Syn, t_fine, c_ops, e_ops=[sx1, sx2])
sx1_fine = np.array(result_a.expect[0])
sx2_fine = np.array(result_a.expect[1])

Delta_t   = 7.0
win       = int(round(Delta_t / dt_fine))   # 70 fine-grid points per window

def pearson_window(s1, s2, i, w):
    """Pearson coefficient of s1, s2 over [i, i+w]."""
    a, b   = s1[i:i+w], s2[i:i+w]
    da, db = a - a.mean(), b - b.mean()
    denom  = np.sqrt((da**2).sum() * (db**2).sum())
    return float((da*db).sum() / denom) if denom > 1e-20 else 0.0

# Evaluate at every 1/ω₁ (every 10 fine steps)
stride      = 10
t_eval      = t_fine[::stride]
pearson_arr = np.array([pearson_window(sx1_fine, sx2_fine, i*stride, win)
                         for i in range(len(t_eval))])
print(f"  Late-time Pearson mean: {pearson_arr[-200:].mean():.3f}  (expect about -1)")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Panel (b) — Excited-state populations  ⟨P^e_k(t)⟩
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel (b): evolving populations…")
t_sub    = np.linspace(0, 1000.0, 500)
result_b = qt.mesolve(H, rho_sub, t_sub, c_ops, e_ops=[Pe1, Pe2])
Pe1_arr  = np.array(result_b.expect[0])
Pe2_arr  = np.array(result_b.expect[1])

# ═══════════════════════════════════════════════════════════════════════════════
# 11. Panel (c) — Negativity  N(t)  and Collectiveness  Ī(ε(t))
# ═══════════════════════════════════════════════════════════════════════════════
print("Panel (c): negativity and collectiveness…")
t_neg    = np.linspace(0, 100.0, 150)
result_c = qt.mesolve(H, rho_C, t_neg, c_ops)

# -- Negativity via qt.negativity(rho, subsys) --
neg_arr = np.array([qt.negativity(s, 1) for s in result_c.states])
N_M     = float(neg_arr.max())
t_NM    = float(t_neg[neg_arr.argmax()])
print(f"  N_M = {N_M:.4f} at t = {t_NM:.1f}/omega_1   (paper: 0.37 at t = 21)")

# -- Collectiveness via Choi state (Eqs. 14-16) --
#    Φ(t) = (ε(t) ⊗ I)[|Ψ⟩⟨Ψ|]   where |Ψ⟩ = ½ Σ_{jk} |jk⟩|jk⟩
#    We build Φ from the propagator using qt.propagator, then compute the
#    normalised mutual information of the 16×16 Choi matrix.

def collectiveness_at(t):
    """Compute Ī(ε(t)) from the Choi state of the quantum map ε(t)."""
    # Propagator as 16×16 numpy array
    U = qt.propagator(H, t, c_ops).full()    # qt.propagator returns the superoperator

    # Build Choi matrix:  Φ_{(IS,ISp),(JS,JSp)} = ¼ U[IS*4+JS, ISp*4+JSp]
    n = 4
    Phi = np.zeros((n**2, n**2), dtype=complex)
    for IS in range(n):
        for JS in range(n):
            for ISp in range(n):
                for JSp in range(n):
                    Phi[IS*n+ISp, JS*n+JSp] = 0.25 * U[IS*n+JS, ISp*n+JSp]

    def vN(M):
        ev = np.linalg.eigvalsh(M)
        ev = ev[ev > 1e-15]
        return float(-np.sum(ev * np.log(ev)))

    # Trace over (q1, q1') → reduced state on (q2, q2')
    rho_22p = np.zeros((n, n), dtype=complex)
    for q2 in range(2):
        for q2p in range(2):
            for q2b in range(2):
                for q2pb in range(2):
                    rho_22p[q2*2+q2p, q2b*2+q2pb] = sum(
                        Phi[8*q1+4*q2+2*q1p+q2p, 8*q1+4*q2b+2*q1p+q2pb]
                        for q1 in range(2) for q1p in range(2))

    # Trace over (q2, q2') → reduced state on (q1, q1')
    rho_11p = np.zeros((n, n), dtype=complex)
    for q1 in range(2):
        for q1p in range(2):
            for q1b in range(2):
                for q1pb in range(2):
                    rho_11p[q1*2+q1p, q1b*2+q1pb] = sum(
                        Phi[8*q1+4*q2+2*q1p+q2p, 8*q1b+4*q2+2*q1pb+q2p]
                        for q2 in range(2) for q2p in range(2))

    return (vN(rho_11p) + vN(rho_22p) - vN(Phi)) / (4 * np.log(2))

t_coll   = np.unique(np.concatenate([np.linspace(0, 20, 10),
                                      np.linspace(20, 60, 10),
                                      np.linspace(60, 100, 6)]))
print(f"  Computing {len(t_coll)} Choi states…")
coll_arr = np.array([collectiveness_at(t) for t in t_coll])
I_M      = float(coll_arr.max())
t_IM     = float(t_coll[coll_arr.argmax()])
print(f"  I_M = {I_M:.4f} at t = {t_IM:.1f}/omega_1   (paper: 0.81)")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. Plotting — white background, publication style
# ═══════════════════════════════════════════════════════════════════════════════
BLUE   = '#1f77b4'
ORANGE = '#ff7f0e'
RED    = '#d62728'
CYAN   = '#17becf'
PURPLE = '#9467bd'
BLACK  = 'black'

plt.rcParams.update({
    'font.family':           'serif',
    'font.size':             11,
    'axes.linewidth':        0.8,
    'xtick.direction':       'in',
    'ytick.direction':       'in',
    'xtick.top':             True,
    'ytick.right':           True,
    'xtick.minor.visible':   True,
    'ytick.minor.visible':   True,
    'figure.facecolor':      'white',
    'axes.facecolor':        'white',
    'savefig.facecolor':     'white',
    'savefig.dpi':           200,
    'savefig.bbox':          'tight',
})

# ── Panel (a): Pearson coefficient ────────────────────────────────────────────
fig_a, ax_a = plt.subplots(figsize=(7, 4.5))

ax_a.plot(t_eval, pearson_arr, color=BLACK, lw=0.8)
ax_a.axhline(0,  color='gray', lw=0.5, ls='--', alpha=0.6)
ax_a.axhline(-1, color='gray', lw=0.4, ls=':',  alpha=0.4)
ax_a.set_xlim(0, 1000)
ax_a.set_ylim(-1.1, 0.65)
ax_a.set_xlabel(r'$t\omega_1$', fontsize=12)
ax_a.set_ylabel(r'$\mathcal{C}_{\Delta t}(t)$', fontsize=12)
ax_a.set_title('(a)', loc='left', fontweight='bold')

# Inset — late-time synchronised oscillations [800, 850]
ax_i1 = ax_a.inset_axes([0.50, 0.6, 0.43, 0.33])
m1 = (t_fine >= 800) & (t_fine <= 850)
ax_i1.plot(t_fine[m1], sx1_fine[m1], color=BLUE,   lw=1.2,
           label=r'$\langle\sigma_1^x(t)\rangle$')
ax_i1.plot(t_fine[m1], sx2_fine[m1], color=ORANGE, lw=1.2,
           label=r'$\langle\sigma_2^x(t)\rangle$')
ax_i1.set_xlim(800, 850)
ax_i1.tick_params(labelsize=8)
ax_i1.set_xlabel(r'$t\omega_1$', fontsize=8, labelpad=1)
ax_i1.legend(fontsize=7, loc='upper right', handlelength=1.2,
             framealpha=0.9, edgecolor='lightgray')
for sp in ax_i1.spines.values(): sp.set_linewidth(0.6)

# Inset — early incoherent transient [100, 150]
ax_i2 = ax_a.inset_axes([0.50, 0.22, 0.43, 0.33])
m2 = (t_fine >= 100) & (t_fine <= 150)
ax_i2.plot(t_fine[m2], sx1_fine[m2], color=BLUE,   lw=1.2)
ax_i2.plot(t_fine[m2], sx2_fine[m2], color=ORANGE, lw=1.2)
ax_i2.set_xlim(100, 150)
ax_i2.tick_params(labelsize=8)
ax_i2.set_xlabel(r'$t\omega_1$', fontsize=8, labelpad=1)
for sp in ax_i2.spines.values(): sp.set_linewidth(0.6)

fig_a.tight_layout()
fig_a.savefig('fig5a_pearson.png')
print("Saved: fig5a_pearson.png")

# ── Panel (b): Excited-state populations ──────────────────────────────────────
fig_b, ax_b = plt.subplots(figsize=(6, 4.5))

ax_b.plot(t_sub, Pe1_arr, color=RED,  lw=1.5, label=r'$\langle P_1^e(t)\rangle$')
ax_b.plot(t_sub, Pe2_arr, color=BLUE, lw=1.5, label=r'$\langle P_2^e(t)\rangle$')
ax_b.set_xlim(0, 1000)
ax_b.set_ylim(0.40, 0.62)
ax_b.set_xlabel(r'$t\omega_1$', fontsize=12)
ax_b.set_ylabel('Population', fontsize=12)
ax_b.set_title('(b)', loc='left', fontweight='bold')
ax_b.legend(fontsize=10, loc='upper right', framealpha=0.9, edgecolor='lightgray')

fig_b.tight_layout()
fig_b.savefig('fig5b_subradiance.png')
print("Saved: fig5b_subradiance.png")

# ── Panel (c): Negativity and Collectiveness ───────────────────────────────────
fig_c, ax_c = plt.subplots(figsize=(6, 4.5))

ax_c.plot(t_neg,  neg_arr,  color=CYAN,   lw=2.0, label=r'$\mathcal{N}(t)$')
ax_c.plot(t_coll, coll_arr, color=PURPLE, lw=2.0,
          label=r'$\bar{I}(\mathcal{E}(t))$')
ax_c.annotate(f'$\\mathcal{{N}}_M = {N_M:.2f}$',
              xy=(t_NM, N_M), xytext=(t_NM+6, N_M+0.05),
              fontsize=10, color=CYAN,
              arrowprops=dict(arrowstyle='->', color=CYAN, lw=1.0))
ax_c.annotate(f'$\\bar{{I}}_M = {I_M:.2f}$',
              xy=(t_IM, I_M), xytext=(t_IM+6, I_M+0.05),
              fontsize=10, color=PURPLE,
              arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.0))
ax_c.set_xlim(0, 100)
ax_c.set_ylim(-0.02, 1.0)
ax_c.set_xlabel(r'$t\omega_1$', fontsize=12)
ax_c.set_ylabel('Value', fontsize=12)
ax_c.set_title('(c)', loc='left', fontweight='bold')
ax_c.legend(fontsize=10, loc='upper right', framealpha=0.9, edgecolor='lightgray')

fig_c.tight_layout()
fig_c.savefig('fig5c_entanglement.png')
print("Saved: fig5c_entanglement.png")

print(f"\nDone.  N_M = {N_M:.4f},  I_M = {I_M:.4f}")