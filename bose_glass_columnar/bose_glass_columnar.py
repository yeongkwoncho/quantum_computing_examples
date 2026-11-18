"""Vortex lines pinned by columnar defects -> Bose glass (Nelson-Vinokur mapping).

A type-II superconductor with columnar defects (e.g. heavy-ion tracks) along the
field direction z.  The partition function of N_v vortex lines is the imaginary-time
path integral of N_v 2D quantum bosons:
    vortex line  <-> boson worldline       z (thickness L) <-> imaginary time (beta)
    temperature T <-> hbar                 columnar defect <-> static random well -V
    B / B_phi    <-> filling N_v / N_d     field tilt H_perp <-> twist theta (vector potential)
    tilt modulus c44 <-> 1 / rho_s         vortex liquid <-> superfluid,  pinned <-> Bose glass
Lattice model (hard-core bosons = XY qubits, |1> = vortex present):
    H = -t sum_<ij> (e^{i phi_ij} b_i^+ b_j + h.c.) + sum_i eps_i n_i
    eps_i = background point disorder U[-0.25, 0.25]t, minus V*(0.8..1.2) on the defect sites

- Classical ED (the "existing method" baseline here): 4x4 torus, 4 defects, N_v = 1..7,
  disorder-averaged superfluid stiffness rho_s = 2 dE(theta)/theta^2 and charge gap
- Quantum circuits (Qiskit + PennyLane, 8-qubit 2x4 ladder): adiabatic state preparation
  from a pinned product state, then a slow field tilt (twist ramp 0 -> pi/2);
  twist energy compared with ED and with QuTiP time-dependent sesolve
- Single tilted line (Hatano-Nelson): non-Hermitian depinning by transverse field
Saves bose_glass_columnar.png next to this script
"""
import time
from pathlib import Path
from math import comb
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, expm, eigvals
import qutip as qt

T_HOP, BG, TH_ED = 1.0, 0.25, 0.3

# ---------------- lattice + sector ED (bit i of the integer = site i = qubit i) ----------------
def lattice(Lx, Ly, pbc_y=True):
    idx = lambda x, y: x + Lx * y
    bonds = []                                   # (i, j, dx): b_i^+ b_j carries phase e^{i dx phi}
    for y in range(Ly):
        for x in range(Lx):
            bonds.append((idx((x + 1) % Lx, y), idx(x, y), 1))
            if pbc_y or y + 1 < Ly:
                bonds.append((idx(x, (y + 1) % Ly), idx(x, y), 0))
    return bonds

def sector(Ns, N):
    s = np.arange(2 ** Ns, dtype=np.int64)
    pop = np.array([bin(v).count("1") for v in range(2 ** Ns)]) if Ns <= 16 else None
    if pop is None:
        from itertools import combinations
        return np.array(sorted(sum(1 << i for i in c) for c in combinations(range(Ns), N)), dtype=np.int64)
    return s[pop == N]

def K_sector(Ns, N, bonds, Lx, theta, states=None):
    states = sector(Ns, N) if states is None else states
    phi = theta / Lx
    rows, cols, vals = [], [], []
    ar = np.arange(len(states))
    for i, j, dx in bonds:
        for a, b, ph in ((i, j, phi * dx), (j, i, -phi * dx)):          # b_a^+ b_b
            m = ((states >> b) & 1).astype(bool) & ~((states >> a) & 1).astype(bool)
            s2 = states[m] ^ ((1 << a) | (1 << b))
            rows.append(np.searchsorted(states, s2)); cols.append(ar[m])
            vals.append(np.full(m.sum(), -T_HOP * np.exp(1j * ph)))
    D = len(states)
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D))
    occ = ((states[:, None] >> np.arange(Ns)) & 1).astype(float)
    return K, occ, states

def ground(K, occ, eps):
    H = K + sp.diags(occ @ eps)
    if H.shape[0] <= 64:
        return eigh(H.toarray(), eigvals_only=True)[0]
    return eigsh(H, k=1, which="SA")[0][0]

def realization(Ns, Nd, V, rng):
    defects = rng.choice(Ns, Nd, replace=False)
    eps = rng.uniform(-BG, BG, Ns)
    eps[defects] -= V * rng.uniform(0.8, 1.2, Nd)
    return eps, defects

# ---------------- (1) phase diagram on the 4x4 torus ----------------
t0 = time.time()
LX, LY, NS, ND, REAL = 4, 4, 16, 4, 30
bonds44 = lattice(LX, LY)
NV_LIST = list(range(0, 9))
Kc = {(N, th): K_sector(NS, N, bonds44, LX, th)[:2] for N in NV_LIST for th in (0.0, TH_ED) if 0 < N}
V_LIST = np.arange(0, 12.1, 1.0)
E0 = np.zeros((len(V_LIST), REAL, 9)); ET = np.zeros_like(E0)
for a, V in enumerate(V_LIST):
    for r in range(REAL):
        eps, _ = realization(NS, ND, V, np.random.default_rng(100 + r))
        for N in range(1, 9):
            E0[a, r, N] = ground(*Kc[(N, 0.0)], eps)
            if N <= 7:
                ET[a, r, N] = ground(*Kc[(N, TH_ED)], eps)
NV = np.arange(1, 8)
rho = 2 * (ET[:, :, 1:8] - E0[:, :, 1:8]) / TH_ED ** 2                  # (V, real, N)
rho_m, rho_s = rho.mean(1), rho.std(1) / np.sqrt(REAL)
frac = rho_m / rho_m[0]
gap = E0[:, :, 2:9] + E0[:, :, 0:7] - 2 * E0[:, :, 1:8]                 # charge gap for N = 1..7
gap_m = gap.mean(1)
print(f"[ED 4x4] {len(V_LIST)} V x {REAL} disorders x 8 sectors in {time.time() - t0:.1f}s")
for N in (2, 4, 6):
    print(f"  B/B_phi={N / ND:.2f}: rho_s(V=0)={rho_m[0, N - 1]:.3f}  rho_s(V=12)={rho_m[-1, N - 1]:.4f}"
          f"  gap(V=8)={gap_m[8, N - 1]:.2f}")

# ---------------- (2) quantum circuits on a 2x4 ladder (8 qubits) ----------------
LXL, LYL, NQ = 4, 2, 8
bondsL = lattice(LXL, LYL, pbc_y=False)
rngL = np.random.default_rng(3)
DEFL = np.array([0, 2, 5]); DEPTH = rngL.uniform(0.8, 1.2, 3); BGL = rngL.uniform(-BG, BG, NQ)
def eps_ladder(V):
    e = BGL.copy(); e[DEFL] -= V * DEPTH
    return e
NEIGH = {i: set() for i in range(NQ)}
for i, j, _ in bondsL:
    NEIGH[i].add(j); NEIGH[j].add(i)

def warm_start(eps, N):
    """Classical greedy guess: fill the deepest wells, then the free site with most empty neighbours."""
    occ = []
    for _ in range(N):
        free = [i for i in range(NQ) if i not in occ]
        occ.append(min(free, key=lambda i: eps[i] - T_HOP * sum(n not in occ for n in NEIGH[i])))
    return sorted(occ)
def naive_start(eps, N):
    return sorted(np.argsort(eps)[:N].tolist())

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter

def pauli_terms(eps, lam, phi, bias_sites, bias=3.0):
    """Pauli list of H = lam*K(phi) + sum eps n + (1-lam) * bias on warm-start sites (no constant)."""
    terms = []
    for i, j, dx in bondsL:
        c, s = np.cos(phi * dx), np.sin(phi * dx)
        terms += [("XX", [i, j], -lam * T_HOP * c / 2), ("YY", [i, j], -lam * T_HOP * c / 2)]
        if dx and abs(s) > 1e-12:
            terms += [("XY", [i, j], lam * T_HOP * s / 2), ("YX", [i, j], -lam * T_HOP * s / 2)]
    w = eps.copy()
    w[bias_sites] -= (1 - lam) * bias
    terms += [("Z", [i], -w[i] / 2) for i in range(NQ)]                  # sum w n = const - sum w Z / 2
    return [t for t in terms if abs(t[2]) > 1e-12]
def H_op(eps, lam, phi, bias_sites):
    const = (eps.sum() - (1 - lam) * 3.0 * len(bias_sites)) / 2
    return SparsePauliOp.from_sparse_list(pauli_terms(eps, lam, phi, bias_sites) + [("", [], const)], num_qubits=NQ)

# sanity: Pauli Hamiltonian restricted to the N sector == sector ED (twisted, with potential)
for N in (2, 4):
    e = eps_ladder(4.0)
    K, occ, st = K_sector(NQ, N, bondsL, LXL, np.pi / 2)
    full = H_op(e, 1.0, np.pi / 2 / LXL, []).to_matrix()
    assert np.allclose(full[np.ix_(st, st)], (K + sp.diags(occ @ e)).toarray()), "Pauli map mismatch"

T_PREP, T_TILT, DT, TH_C = 30.0, 15.0, 0.05, np.pi / 2
N_PREP, N_TILT = int(round(T_PREP / DT)), int(round(T_TILT / DT))
def schedule(tm):
    """lambda(t) ramps hopping on, then theta(t) tilts the field; returns (lam, phi per bond)."""
    if tm < T_PREP:
        return np.sin(np.pi * tm / (2 * T_PREP)) ** 2, 0.0
    return 1.0, TH_C / LXL * np.sin(np.pi * (tm - T_PREP) / (2 * T_TILT)) ** 2
STEP_T = [(k + 0.5) * DT for k in range(N_PREP + N_TILT)]

def exact_twist(eps, N):
    K0, occ, _ = K_sector(NQ, N, bondsL, LXL, 0.0); KT = K_sector(NQ, N, bondsL, LXL, TH_C)[0]
    return ground(K0, occ, eps), ground(KT, occ, eps)

# (2a) adiabatic fidelity, naive vs warm start (sector propagation, exact exponentials)
def sector_adiabatic(eps, N, start):
    K, occ, st = K_sector(NQ, N, bondsL, LXL, 0.0); K = K.toarray(); D = np.diag(occ @ eps)
    k0 = int(np.flatnonzero((occ[:, start].sum(1) == N))[0])
    B = np.diag(-3.0 * occ @ occ[k0])
    psi = np.zeros(len(st), complex); psi[k0] = 1
    for tm in STEP_T[:N_PREP]:
        lam = schedule(tm)[0]
        psi = expm(-1j * DT * (lam * K + D + (1 - lam) * B)) @ psi
    w, v = eigh(K + D)
    return abs(v[:, 0].conj() @ psi) ** 2, w[1] - w[0]
V_FID = np.arange(0, 8.1, 1.0)
fid = {(N, m): [sector_adiabatic(eps_ladder(V), N, f(eps_ladder(V), N))[0] for V in V_FID]
       for N in (2, 4) for m, f in (("naive", naive_start), ("warm", warm_start))}

# (2b) circuits
V_C = [0.0, 2.0, 4.0, 6.0, 8.0]
N_C = (2, 4)
def qiskit_run(eps, N):
    sites = warm_start(eps, N)
    qc = QuantumCircuit(NQ); [qc.x(i) for i in sites]
    sv = Statevector(qc); out = {}
    for k, tm in enumerate(STEP_T):
        lam, phi = schedule(tm)
        step = QuantumCircuit(NQ)
        step.append(PauliEvolutionGate(SparsePauliOp.from_sparse_list(pauli_terms(eps, lam, phi, sites), NQ),
                                       time=DT, synthesis=SuzukiTrotter(order=2, reps=1)), range(NQ))
        sv = sv.evolve(step.decompose(reps=3))
        if k == N_PREP - 1:
            out["prep"] = sv
    out["tilt"] = sv
    return out, sites

import pennylane as qml
devL = qml.device("default.qubit", wires=NQ)
def pl_H(eps, lam, phi, sites):
    P = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    co, ops = [], []
    for p_, w_, c in pauli_terms(eps, lam, phi, sites):
        co.append(c); ops.append(qml.prod(*[P[ch](w) for ch, w in zip(p_, w_)]) if len(w_) > 1 else P[p_](w_[0]))
    return qml.Hamiltonian(co, ops)
@qml.qnode(devL)
def pl_circuit(eps, sites, nsteps):
    for i in sites:
        qml.PauliX(i)
    for tm in STEP_T[:nsteps]:
        lam, phi = schedule(tm)
        qml.TrotterProduct(pl_H(eps, lam, phi, sites), time=DT, n=1, order=2)
    return qml.state()

def qutip_run(eps, N, sites):
    """Reference: continuous-time Schroedinger equation with the same schedule (Qiskit qubit order)."""
    M = lambda ops: qt.Qobj(SparsePauliOp.from_sparse_list(ops, NQ).to_matrix()) if ops else 0 * qt.qeye(2 ** NQ)
    Ky = M([(p, [i, j], -T_HOP / 2) for i, j, dx in bondsL if dx == 0 for p in ("XX", "YY")])
    KxA = M([(p, [i, j], -T_HOP / 2) for i, j, dx in bondsL if dx == 1 for p in ("XX", "YY")])
    KxB = M([("XY", [i, j], T_HOP / 2) for i, j, dx in bondsL if dx == 1] + [("YX", [i, j], -T_HOP / 2) for i, j, dx in bondsL if dx == 1])
    D = M([("Z", [i], -eps[i] / 2) for i in range(NQ)])
    Bz = M([("Z", [i], 3.0 / 2) for i in sites])                          # -(3) n_i  ->  +3/2 Z_i (+const)
    lam = lambda t: schedule(t)[0]
    H = [D, [Ky, lam], [KxA, lambda t: lam(t) * np.cos(schedule(t)[1])], [KxB, lambda t: lam(t) * np.sin(schedule(t)[1])],
         [Bz, lambda t: 1 - lam(t)]]
    psi0 = qt.Qobj(np.eye(2 ** NQ)[sum(1 << i for i in sites)].reshape(-1, 1))
    res = qt.sesolve(H, psi0, [0, T_PREP, T_PREP + T_TILT], options={"atol": 1e-10, "rtol": 1e-8, "nsteps": 100000})
    return res.states[1].full().ravel(), res.states[2].full().ravel()

t0 = time.time()
circ = {}
for N in N_C:
    for V in V_C:
        e = eps_ladder(V)
        q, sites = qiskit_run(e, N)
        Hp, Ht = H_op(e, 1.0, 0.0, []), H_op(e, 1.0, TH_C / LXL, [])
        Hp_pl, Ht_pl = (qml.matrix(pl_H(e, 1.0, ph, []), wire_order=range(NQ)) for ph in (0.0, TH_C / LXL))
        cst = e.sum() / 2
        s_pp, s_pt = pl_circuit(e, sites, N_PREP), pl_circuit(e, sites, N_PREP + N_TILT)
        r_p, r_t = qutip_run(e, N, sites)
        Mp, Mt = Hp.to_matrix(), Ht.to_matrix()
        nq = lambda sv: np.array([sv.probabilities([i])[1] for i in range(NQ)])
        npl = lambda s: np.array([np.sum(np.abs(s.reshape([2] * NQ)).take(1, axis=i) ** 2) for i in range(NQ)])
        nqt = lambda s: np.array([np.sum(np.abs(s) ** 2 * ((np.arange(2 ** NQ) >> i) & 1)) for i in range(NQ)])
        ex0, exT = exact_twist(e, N)
        K0, occ, st = K_sector(NQ, N, bondsL, LXL, 0.0)
        w, v = eigh((K0 + sp.diags(occ @ e)).toarray())
        circ[(N, V)] = dict(
            qk=(np.real(q["tilt"].expectation_value(Ht) - q["prep"].expectation_value(Hp)), nq(q["prep"])),
            pl=(np.real(s_pt.conj() @ Ht_pl @ s_pt - s_pp.conj() @ Hp_pl @ s_pp), npl(s_pp)),
            qt=(np.real(r_t.conj() @ Mt @ r_t - r_p.conj() @ Mp @ r_p), nqt(r_p)),
            exact=(exT - ex0, np.abs(v[:, 0]) ** 2 @ occ),
            states=(q["prep"], q["tilt"], Hp, Ht))
        c = circ[(N, V)]
        print(f"  ladder N_v={N} V={V:g}: dE_twist exact {c['exact'][0]:.4f}  QuTiP {c['qt'][0]:.4f}"
              f"  Qiskit {c['qk'][0]:.4f}  PennyLane {c['pl'][0]:.4f}")
print(f"[circuits] {len(circ)} adiabatic runs x {N_PREP + N_TILT} Trotter steps in {time.time() - t0:.1f}s")
err_trot = max(max(abs(c["qk"][0] - c["qt"][0]), abs(c["pl"][0] - c["qt"][0]),
                   np.abs(c["qk"][1] - c["qt"][1]).max(), np.abs(c["pl"][1] - c["qt"][1]).max()) for c in circ.values())
err_gs = max(abs(c["qk"][0] - c["exact"][0]) for c in circ.values())
print(f"  max |circuit - QuTiP| (twist energy, densities) = {err_trot:.2e};  max |circuit - ED| twist energy = {err_gs:.2e}")

# shots to resolve the twist energy at 2 sigma (each Pauli term measured separately, shots split evenly)
def shots_needed(c):
    sp_, st_, Hp, Ht = c["states"]
    var = 0.0
    for sv, H in ((sp_, Hp), (st_, Ht)):
        terms = [(P, co) for P, co in zip(H.paulis, H.coeffs) if P.to_label().strip("I")]
        var += len(terms) * sum(abs(co) ** 2 * (1 - np.real(sv.expectation_value(SparsePauliOp(P))) ** 2) for P, co in terms)
    return 4 * var / c["exact"][0] ** 2
shots = {(N, V): shots_needed(circ[(N, V)]) for N in N_C for V in V_C}

# ---------------- (3) single tilted line: Hatano-Nelson ----------------
NHN, WHN, RHN = 200, 4.0, 20
G_LIST = np.linspace(0, 2.6, 27)
def hn(g, e):
    return np.diag(e) - T_HOP * (np.exp(g) * np.eye(NHN, k=-1) + np.exp(-g) * np.eye(NHN, k=1)
                                 + np.exp(g) * np.eye(NHN, k=NHN - 1) + np.exp(-g) * np.eye(NHN, k=-(NHN - 1)))
cfrac, gs_im = [], []
for g in G_LIST:
    f, gi = 0, 0
    for r in range(RHN):
        e = np.random.default_rng(r).uniform(-WHN / 2, WHN / 2, NHN)
        ev = eigvals(hn(g, e)); f += np.mean(np.abs(ev.imag) > 1e-6); gi += np.abs(ev.imag[np.argmin(ev.real)]) > 1e-6
    cfrac.append(f / RHN); gs_im.append(gi / RHN)
e_show = np.random.default_rng(0).uniform(-WHN / 2, WHN / 2, NHN)
spec = {g: eigvals(hn(g, e_show)) for g in (0.0, 0.5, 1.0)}

# ---------------- (4) classical ED cost (measured) ----------------
cost = []
for lx, ly in ((4, 3), (4, 4), (5, 4), (6, 4), (7, 4)):
    ns, nv = lx * ly, lx * ly // 4
    t0 = time.time()
    K, occ, _ = K_sector(ns, nv, lattice(lx, ly), lx, 0.0)
    eps, _ = realization(ns, nv, 6.0, np.random.default_rng(0))
    ground(K, occ, eps)
    cost.append((ns, comb(ns, nv), time.time() - t0))
    print(f"  ED {lx}x{ly}: dim {comb(ns, nv)}  time {cost[-1][2]:.2f}s")

ok = err_trot < 1e-2 and err_gs < 2e-2
print("PASS" if ok else "FAIL", "(circuits vs QuTiP < 1e-2, adiabatic twist energy vs ED < 2e-2)")

# ======================= figure =======================
import sys; sys.path.insert(0, str(Path(__file__).parent))
from _style import *
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Polygon

fig = plt.figure(figsize=(17, 20.5))
gs = fig.add_gridspec(4, 3, hspace=0.42, wspace=0.32, top=0.95)
FCOL = {2: VIOLET, 4: AQUA, 6: MAGENTA}

# (a) 3D schematic
ax = fig.add_subplot(gs[0, 0]); ax.set_aspect("equal"); ax.axis("off")
pj = lambda x, y, z: (x + 0.5 * y, z + 0.32 * y)
box = [pj(0, 0, 0), pj(6, 0, 0), pj(6, 3, 0), pj(6, 3, 5), pj(0, 3, 5), pj(0, 0, 5)]
ax.add_patch(Polygon([pj(0, 0, 0), pj(6, 0, 0), pj(6, 3, 0), pj(0, 3, 0)], fc="#eef3fb", ec=MUTED, lw=1))
for a_, b_ in (((0, 0, 0), (0, 0, 5)), ((6, 0, 0), (6, 0, 5)), ((6, 3, 0), (6, 3, 5)), ((0, 3, 0), (0, 3, 5)),
               ((0, 0, 5), (6, 0, 5)), ((6, 0, 5), (6, 3, 5)), ((6, 3, 5), (0, 3, 5)), ((0, 3, 5), (0, 0, 5))):
    ax.plot(*zip(pj(*a_), pj(*b_)), color=MUTED, lw=1)
cols_xy = [(1.0, 0.8), (2.6, 2.2), (4.2, 1.0), (5.2, 2.4)]
for x, y in cols_xy:
    (u0, v0), (u1, v1) = pj(x, y, 0), pj(x, y, 5)
    ax.add_patch(Rectangle((u0 - 0.13, v0), 0.26, v1 - v0, fc="#c9c8c3", ec="none", alpha=0.9))
zz = np.linspace(0, 5, 60)
for x, y in cols_xy[:3]:
    ax.plot(*zip(*[pj(x, y, z) for z in zz]), color=VIOLET, lw=2.2)
rw = np.random.default_rng(4)
for x0, y0 in ((3.3, 0.4), (1.8, 2.6)):
    xs = x0 + np.cumsum(rw.normal(0, 0.09, 60)); ys = y0 + np.cumsum(rw.normal(0, 0.06, 60))
    ax.plot(*zip(*[pj(a, b, z) for a, b, z in zip(xs, ys, zz)]), color=ORANGE, lw=2.2)
ax.annotate("", xy=pj(-0.6, 0, 5), xytext=pj(-0.6, 0, 0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
ax.text(*pj(-0.9, 0, 2.5), "B ∥ z\n≡ imaginary\ntime τ", ha="right", va="center", fontsize=9)
ax.text(*pj(3, 0, -0.55), "pinned on column (Bose glass)", color=VIOLET, ha="center", fontsize=9)
ax.text(*pj(3, 0, -1.15), "wandering line (vortex liquid)", color=ORANGE, ha="center", fontsize=9)
ax.text(*pj(6.2, 3, 5.15), "columnar defects", color=MUTED, ha="right", fontsize=9)
ax.set_xlim(-3.2, 8.2); ax.set_ylim(-1.6, 6.4)
ax.set_title("Superconductor with columnar defects", loc="left")

# (b) mapping: 2D boson lattice + dictionary
ax = fig.add_subplot(gs[0, 1]); ax.set_aspect("equal"); ax.axis("off")
eps_ex, def_ex = realization(NS, ND, 6.0, np.random.default_rng(100))
for i, j, dx in bonds44:
    xi, yi, xj, yj = i % LX, i // LX, j % LX, j // LX
    if abs(xi - xj) <= 1 and abs(yi - yj) <= 1:
        ax.plot([xi, xj], [yi, yj], color=AQUA, lw=1.5, zorder=1)
for s in range(NS):
    x, y = s % LX, s // LX
    ax.add_patch(Circle((x, y), 0.24 if s in def_ex else 0.13, fc="#9b9a95" if s in def_ex else "#e6e5e1", ec=INK, lw=1, zorder=2))
for s in sorted(def_ex, key=lambda s: eps_ex[s])[:3]:
    ax.add_patch(Circle((s % LX, s // LX), 0.12, fc=VIOLET, ec="none", zorder=3))
ax.annotate("", xy=(3.75, -0.55), xytext=(-0.25, -0.55), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
ax.text(1.75, -0.85, "twist θ across x  ≡  tilt the field", color=ORANGE, ha="center", va="top", fontsize=9)
ax.text(4.3, 1.5, "grey: defect\nwell −V\n\npurple:\nvortex =\nboson", fontsize=8.5, color=MUTED, ha="left", va="center")
dic = ["vortex line  ↔  boson worldline", "thickness L  ↔  β (1/temperature)", "temperature T  ↔  ħ",
       "B / B_φ  ↔  filling N_v / N_d", "tilt modulus c₄₄  ↔  1/ρ_s",
       "vortex liquid  ↔  superfluid (ρ_s > 0)", "Bose glass  ↔  Bose glass (ρ_s = 0)", "B = B_φ  ↔  Mott insulator"]
for k, line in enumerate(dic):
    ax.text(-0.3, -1.6 - 0.42 * k, line, fontsize=9, va="top", family="DejaVu Sans")
ax.set_xlim(-0.6, 5.6); ax.set_ylim(-5.2, 3.6)
ax.set_title("Mapping: vortex lines → 2D bosons", loc="left")

# (c) phase diagram heatmap
ax = fig.add_subplot(gs[0, 2]); ax.grid(False)
im = ax.imshow(np.clip(frac.T, 0, 1.05), origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1,
               extent=(V_LIST[0] - 0.5, V_LIST[-1] + 0.5, NV[0] / ND - 0.125, NV[-1] / ND + 0.125))
ax.axhline(1.0, color=ORANGE, lw=1.2, ls="--")
ax.text(11.8, 1.06, "matching field B = B_φ", color=ORANGE, ha="right", va="bottom", fontsize=8.5)
ax.text(10.5, 0.42, "Bose glass", color=INK, ha="center", fontsize=10, fontweight="bold")
ax.text(1.8, 1.4, "vortex liquid", color="white", ha="center", fontsize=10, fontweight="bold")
ax.text(10.5, 1.55, "liquid of\nextra vortices", color=INK, ha="center", fontsize=9)
ax.set_xlabel("pinning depth V / t"); ax.set_ylabel("B / B_φ  (vortices per defect)")
ax.set_title(f"Phase diagram (4×4 ED, {REAL} disorders)", loc="left")
fig.colorbar(im, ax=ax, fraction=0.046).set_label("stiffness fraction ρ_s(V) / ρ_s(0)")

# (d) stiffness vs V
ax = fig.add_subplot(gs[1, 0])
for N, col in FCOL.items():
    ax.errorbar(V_LIST, rho_m[:, N - 1], yerr=rho_s[:, N - 1], color=col, lw=2, marker="o", ms=4, capsize=2,
                label=f"B/B_φ = {N / ND:g}")
ax.set_xlabel("pinning depth V / t"); ax.set_ylabel("superfluid stiffness ρ_s (t)")
ax.set_title("Stiffness = inverse tilt modulus (4×4 ED)", loc="left"); ax.legend(loc="upper right")
ax.set_ylim(bottom=0)

# (e) charge gap vs filling
ax = fig.add_subplot(gs[1, 1])
for V, col, mk in ((1.0, AQUA, "s"), (4.0, MUTED, "^"), (8.0, VIOLET, "o")):
    a = int(np.argmin(abs(V_LIST - V)))
    ax.plot(NV / ND, gap_m[a], color=col, lw=2, marker=mk, ms=6, label=f"V = {V:g}t")
ax.axvline(1.0, color=ORANGE, lw=1.2, ls="--")
ax.text(1.04, ax.get_ylim()[1] * 0.92, "B = B_φ:\nMott gap\n(all columns full)", color=ORANGE, fontsize=8.5, va="top")
ax.set_xlabel("B / B_φ"); ax.set_ylabel("charge gap E(N+1)+E(N−1)−2E(N)  (t)")
ax.set_title("Compressibility: gap only at matching", loc="left"); ax.legend(loc="upper left")

# (f) Hatano-Nelson spectra
ax = fig.add_subplot(gs[1, 2])
for g, col, mk in ((0.0, MUTED, "o"), (0.5, AQUA, "s"), (1.0, ORANGE, "^")):
    ev = spec[g]
    ax.plot(ev.real, ev.imag, mk, ms=3.5, color=col, label=f"tilt g = {g:g}", alpha=0.85)
ax.set_xlabel("Re E (t)"); ax.set_ylabel("Im E (t)")
ax.set_title(f"One tilted line (Hatano-Nelson, N={NHN}, W={WHN:g}t)", loc="left", fontsize=10)
ax.legend(loc="upper right", fontsize=8.5)
ax.set_ylim(-2.6, 3.3)
ax.text(-3.4, 3.2, "real E = pinned (transverse Meissner)\ncomplex E = depinned, tilted lines", fontsize=8.5, color=MUTED, va="top")

# (g) adiabatic fidelity
ax = fig.add_subplot(gs[2, 0])
for (N, m), f in fid.items():
    ax.plot(V_FID, f, color=FCOL[N], lw=2, ls="-" if m == "warm" else ":", marker="o" if m == "warm" else "x", ms=6,
            label=f"B/B_φ={N / 3:.2f}, {'greedy warm start' if m == 'warm' else 'lowest-energy start'}")
ax.set_xlabel("pinning depth V / t"); ax.set_ylabel("|⟨GS|ψ(T)⟩|²")
ax.set_ylim(-0.03, 1.12); ax.legend(loc="lower left", fontsize=8.5)
ax.set_title(f"Adiabatic state preparation (T = {T_PREP:g}/t, 8 qubits)", loc="left", fontsize=10)

# (h) density profiles: exact GS vs circuits
ax = fig.add_subplot(gs[2, 1])
xs = np.arange(NQ)
for k, (N, V, col) in enumerate(((2, 0.0, AQUA), (2, 8.0, VIOLET))):
    c = circ[(N, V)]; off = 1.25 * k
    ax.bar(xs, c["exact"][1], 0.6, bottom=off, color=col, alpha=0.45, label=f"exact GS, V = {V:g}t")
    qk_marks(ax, xs, c["qk"][1] + off, label="Qiskit" if k == 0 else None)
    pl_marks(ax, xs, c["pl"][1] + off, label="PennyLane" if k == 0 else None)
    ax.axhline(off, color=GRID, lw=1)
for d in DEFL:
    ax.text(d, 2.42, "▼", ha="center", color=MUTED, fontsize=9)
ax.text(7.4, 2.42, "▼ defect site", ha="right", color=MUTED, fontsize=8.5)
ax.set_xticks(xs); ax.set_xlabel("ladder site (x + 4y)"); ax.set_ylabel("⟨nᵢ⟩  (V = 8t panel offset by 1.25)")
ax.set_ylim(0, 3.25); ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.set_title("Vortex density after adiabatic prep (2 vortices)", loc="left", fontsize=10)

# (i) twist energy, circuits vs ED
ax = fig.add_subplot(gs[2, 2])
for N in N_C:
    ex = [circ[(N, V)]["exact"][0] for V in V_C]
    ax.plot(V_C, ex, color=FCOL[N], lw=2, label=f"exact (ED), B/B_φ = {N / 3:.2f}")
    qk_marks(ax, V_C, [circ[(N, V)]["qk"][0] for V in V_C], label="Qiskit" if N == 2 else None)
    pl_marks(ax, V_C, [circ[(N, V)]["pl"][0] for V in V_C], label="PennyLane" if N == 2 else None)
ax.set_yscale("log"); ax.set_xlabel("pinning depth V / t"); ax.set_ylabel("ΔE_twist = E(θ=π/2) − E(0)  (t)")
ax.set_title("Field-tilt response on the quantum circuit", loc="left"); ax.legend(loc="lower left", fontsize=8.5)

# (j) depinning by tilt
ax = fig.add_subplot(gs[3, 0])
ax.plot(G_LIST, cfrac, color=ORANGE, lw=2, marker="o", ms=4, label="fraction of complex eigenvalues")
ax.set_xlabel("tilt g  (∝ transverse field H⊥)"); ax.set_ylabel("fraction (avg over 20 disorders)")
ax.set_title("Tilt-induced depinning transition", loc="left"); ax.legend(loc="center right", fontsize=8.5)
ax.annotate("g ≲ 0.1: every line stays pinned\n(transverse Meissner effect)", xy=(0.1, 0.01), xytext=(0.35, 0.12), arrowprops=dict(arrowstyle="->", color=MUTED), ha="left", fontsize=8.5, color=MUTED)

# (k) shots
ax = fig.add_subplot(gs[3, 1])
for N in N_C:
    ax.semilogy(V_C, [shots[(N, V)] for V in V_C], color=FCOL[N], lw=2, marker="o", label=f"B/B_φ = {N / 3:.2f}")
ax.set_xlabel("pinning depth V / t"); ax.set_ylabel("shots to resolve ΔE_twist at 2σ")
ax.set_title("Measurement cost grows as ρ_s → 0", loc="left"); ax.legend(loc="upper left")

# (l) ED cost
ax = fig.add_subplot(gs[3, 2])
ns_ = np.array([c[0] for c in cost]); tm_ = np.array([c[2] for c in cost])
ax.semilogy(ns_, tm_, "o-", color=INK, lw=2, label="exact diagonalization (measured)")
for n, d, tt in cost:
    ax.annotate(f"dim {d:,}", (n, tt), textcoords="offset points", xytext=(-6, 8) if n > 12 else (6, 6), fontsize=8, ha="right" if n > 12 else "left", color=MUTED)
ax.set_xlabel("lattice sites (quarter filling)"); ax.set_ylabel("ED wall time (s)")
ax.set_xlim(10, 30)
ax.set_title("Classical ED cost (exponential)", loc="left"); ax.legend(loc="upper left")

suptitle(fig, "Columnar defects & the Bose-glass transition — ED · Qiskit · PennyLane · QuTiP")
out = Path(__file__).parent / "bose_glass_columnar.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)
