"""Neutral-atom (Pasqal) analog program: adiabatic sweep into the Z2 Rydberg crystal.

Pasqal's QPUs are analog: atoms in optical tweezers are driven by one global laser
pulse, and the device directly implements

    H(t) = Ω(t)/2 Σ_i X_i − δ(t) Σ_i n_i + Σ_{i<j} C6 / r_ij^6 n_i n_j ,

n = |r⟩⟨r| (Rydberg). Nearby atoms cannot both be excited (Rydberg blockade,
R_b = (C6/Ω)^(1/6)). Sweeping δ from negative to positive drives a chain from all
ground into the antiferromagnetic Z2 crystal r g r g … r (Bernien et al. 2017).

The program is written in Pulser for Pasqal's `AnalogDevice` (a QPU-like spec)
on its pre-calibrated triangular trap layout, so the same sequence is valid for the
cloud emulators and the real QPU. It is checked three ways:

  * Pulser's own emulator (pulser-simulation, QuTiP backend)
  * an independent QuTiP solve of the Hamiltonian above, from the sampled pulses
  * a Qiskit gate circuit (2nd-order Trotter) of the same H(t): what a gate-based
    machine would have to run to emulate one analog pulse

then a noisy run with illustrative hardware noise (SPAM, dephasing, laser and
Doppler noise), a sweep-time scan, a ground-state phase diagram and a cost panel.

Usage:
    python pasqal_rydberg.py                 # local, free, no account
    python pasqal_rydberg.py --mode cloud    # Pasqal Cloud free emulator (EMU_FREE),
                                             # needs a Pasqal Cloud account + project id
Outputs: pasqal_rydberg.png, pasqal_rydberg_sequence.json (abstract repr = the file
you submit), pasqal_rydberg_cloud.json after a cloud run.
"""
import argparse, json, os, sys, time, getpass
from pathlib import Path
import numpy as np
import qutip as qt
from pulser import Sequence, Pulse, NoiseModel
from pulser.waveforms import InterpolatedWaveform
from pulser.devices import AnalogDevice
from pulser.sampler import sample
from pulser_simulation import QutipEmulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp, Statevector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import *  # noqa: E402,F401

DEV = AnalogDevice
C6 = DEV.interaction_coeff                      # rad/µs · µm^6
OMEGA = 2 * np.pi * 1.8                         # rad/µs (device max 2π·2)
D0, DF = 2 * np.pi * 4.0, 2 * np.pi * 4.0       # detuning sweep −D0 → +DF
T_NS = 4000                                     # sweep time (device max 6000 ns)
T_SCAN = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]
T_NOISY = [1000, 2000, 4000, 6000]
SHOTS = 1000
DT_TROT = 0.01                                  # µs, Qiskit Trotter step
NOISE = dict(state_prep_error=0.005, p_false_pos=0.01, p_false_neg=0.05,
             dephasing_rate=0.2, relaxation_rate=0.01, amp_sigma=0.03,
             temperature=50, laser_waist=175)   # illustrative, not a calibration
NOISE_RUNS = 30                                 # noise trajectories (shot-to-shot laser/Doppler draws)


# ------------------------------------------------------------------ program
def register():
    lay = DEV.pre_calibrated_layouts[0]
    traps = lay.traps_dict
    row = sorted([k for k, v in traps.items() if abs(v[1]) < 1e-6], key=lambda k: traps[k][0])
    reg = lay.define_register(*row, qubit_ids=[f"q{i}" for i in range(len(row))])
    return lay, reg


def sweep(reg, T=T_NS):
    seq = Sequence(reg, DEV)
    seq.declare_channel("ryd", "rydberg_global")
    amp = InterpolatedWaveform(T, [1e-9, OMEGA, OMEGA, 1e-9])
    det = InterpolatedWaveform(T, [-D0, -D0, DF, DF])
    seq.add(Pulse(amp, det, 0), "ryd")
    seq.measure("ground-rydberg")
    return seq


def z2_string(N):            # Pulser bitstrings: '1' = Rydberg
    return "".join("1" if i % 2 == 0 else "0" for i in range(N))


def pulser_run(seq, noise=None, sampling=0.25):
    kw = {} if noise is None else {"noise_model": NoiseModel(runs=NOISE_RUNS, samples_per_run=SHOTS // NOISE_RUNS, **noise)}
    sim = QutipEmulator.from_sequence(seq, sampling_rate=sampling, **kw)
    return sim.run()


def density_pulser(res, N):
    """⟨n_i⟩(t) from Pulser's state list (|r⟩ is basis index 0 in Pulser)."""
    r = qt.basis(2, 0)
    ns = [qt.tensor([r * r.dag() if k == i else qt.qeye(2) for k in range(N)]) for i in range(N)]
    out = []
    for st in res.states:
        s = qt.Qobj(st.full(), dims=ns[0].dims if st.isoper else [ns[0].dims[0], [1] * N])
        out.append([qt.expect(o, s) for o in ns])
    return np.array(out), np.asarray(res._sim_times)


# ------------------------------------------------------------------ independent checks
def own_qutip(seq, pos):
    """sesolve of H(t) written from scratch, driven by Pulser's sampled waveforms."""
    s = sample(seq).channel_samples["ryd"]
    a, d = np.asarray(s.amp), np.asarray(s.det)
    tt = np.arange(len(a)) * 1e-3
    N = len(pos)
    r, g = qt.basis(2, 0), qt.basis(2, 1)
    op = lambda o, i: qt.tensor([o if k == i else qt.qeye(2) for k in range(N)])
    n = [op(r * r.dag(), i) for i in range(N)]
    X = [op(r * g.dag() + g * r.dag(), i) for i in range(N)]
    Hint = sum(C6 / np.linalg.norm(pos[i] - pos[j]) ** 6 * n[i] * n[j]
               for i in range(N) for j in range(i + 1, N))
    H = [Hint, [0.5 * sum(X), qt.coefficient(a, tlist=tt)], [-sum(n), qt.coefficient(d, tlist=tt)]]
    out = qt.sesolve(H, qt.tensor([g] * N), tt, options={"atol": 1e-10, "rtol": 1e-8, "nsteps": 100000})
    psi = out.states[-1]
    return np.array([qt.expect(o, psi) for o in n]), psi, a, d


def ham_pauli(pos, om, de):
    """H with n = (I − Z)/2 on qubits (|1⟩ = Rydberg); Qiskit order handled by indices."""
    N = len(pos)
    terms = [("X", [i], om / 2) for i in range(N)]
    zc = np.full(N, de / 2)                              # −δ n = −δ/2 + δ/2 Z
    for i in range(N):
        for j in range(i + 1, N):
            V = C6 / np.linalg.norm(pos[i] - pos[j]) ** 6
            terms.append(("ZZ", [i, j], V / 4))         # V n n = V/4 (1 − Zi − Zj + ZiZj)
            zc[i] -= V / 4; zc[j] -= V / 4
    terms += [("Z", [i], zc[i]) for i in range(N)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=N)


def qiskit_trotter(pos, a, d, dt=DT_TROT):
    N = len(pos)
    T = len(a) * 1e-3
    steps = int(round(T / dt))
    tt = np.arange(len(a)) * 1e-3
    qc = QuantumCircuit(N)
    for k in range(steps):
        tm = (k + 0.5) * dt
        om, de = np.interp(tm, tt, a), np.interp(tm, tt, d)
        qc.append(PauliEvolutionGate(ham_pauli(pos, om, de), dt, synthesis=SuzukiTrotter(order=2, reps=1)), range(N))
    qc = qc.decompose(reps=3)
    sv = Statevector.from_label("0" * N).evolve(qc)
    p = sv.probabilities_dict()
    nz = np.zeros(N)
    for b, pr in p.items():
        for i in range(N):
            nz[i] += pr * (b[N - 1 - i] == "1")
    step = QuantumCircuit(N)
    step.append(PauliEvolutionGate(ham_pauli(pos, OMEGA, 0.0), dt, synthesis=SuzukiTrotter(order=2, reps=1)), range(N))
    cx1 = transpile(step, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1).count_ops().get("cx", 0)
    return nz, p, steps, cx1


# ------------------------------------------------------------------ ground-state phase diagram
def gs_phase_diagram(N=9, nd=31, nr=31):
    """Exact ground state of the open chain (Ω = 1, V_ij = (R_b/(a|i−j|))^6)."""
    idx = np.arange(2 ** N)
    bits = (idx[:, None] >> np.arange(N)) & 1            # bit i = atom i is Rydberg
    flip = idx[:, None] ^ (1 << np.arange(N))
    nn_c = {}
    deltas = np.linspace(-1, 5, nd); rbs = np.linspace(1.0, 2.6, nr)
    SPi = np.zeros((nr, nd)); S3 = np.zeros((nr, nd))
    X = np.zeros((2 ** N, 2 ** N))
    for i in range(N):
        X[idx, flip[:, i]] += 0.5
    nsum = bits.sum(1)
    dij = np.abs(np.arange(N)[:, None] - np.arange(N)[None])
    ph = lambda k: np.cos(k * dij)
    for ir, rb in enumerate(rbs):
        V = np.where(dij > 0, (rb / np.maximum(dij, 1)) ** 6, 0.0)
        Eint = 0.5 * np.einsum("si,ij,sj->s", bits, V, bits)
        for idd, de in enumerate(deltas):
            H = X.copy(); H[idx, idx] += Eint - de * nsum
            w, v = np.linalg.eigh(H)
            p = v[:, 0] ** 2
            n2 = np.einsum("s,si,sj->ij", p, bits, bits)   # open ends pin the crystal,
            SPi[ir, idd] = (ph(np.pi) * n2).sum() / N        # so use the full (not connected)
            S3[ir, idd] = (ph(2 * np.pi / 3) * n2).sum() / N  # structure factor
    return deltas, rbs, SPi, S3


# ------------------------------------------------------------------ cloud (optional)
def cloud_run(seq, shots):
    from pasqal_cloud import PasqalCloudConnection, RemoteEmuFreeBackend
    user = os.environ.get("PASQAL_USERNAME") or input("Pasqal Cloud username (e-mail): ").strip()
    proj = os.environ.get("PASQAL_PROJECT_ID") or input("project id: ").strip()
    pw = getpass.getpass("password (hidden): ")
    conn = PasqalCloudConnection(username=user, password=pw, project_id=proj)
    del pw
    backend = RemoteEmuFreeBackend(seq, conn)
    print("submitting to EMU_FREE ...")
    rr = backend.run(job_params=[{"runs": shots}], wait=True)
    counts = dict(rr.results[0].final_bitstrings)
    json.dump({"batch_id": rr.batch_id, "counts": counts}, open(HERE / "pasqal_rydberg_cloud.json", "w"), indent=1)
    return counts


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "cloud"], default="local")
    ap.add_argument("--shots", type=int, default=SHOTS)
    args = ap.parse_args()
    t_start = time.time(); np.random.seed(0)
    lay, reg = register()
    pos = np.array([np.asarray(p) for p in reg.qubits.values()])
    N = len(pos); a_sp = np.linalg.norm(pos[1] - pos[0]); Rb = DEV.rydberg_blockade_radius(OMEGA)
    seq = sweep(reg)
    DEV.validate_register(reg)
    (HERE / "pasqal_rydberg_sequence.json").write_text(seq.to_abstract_repr())
    print(f"{N} atoms, a = {a_sp:.1f} µm, R_b = {Rb:.2f} µm (R_b/a = {Rb / a_sp:.2f}); "
          f"sequence valid for {DEV.name}, saved pasqal_rydberg_sequence.json")
    Z2 = z2_string(N)

    if args.mode == "cloud":
        counts = cloud_run(seq, args.shots)
        tot = sum(counts.values())
        n_own, psi_own, _, _ = own_qutip(seq, pos)
        amp = psi_own.full().ravel()
        flip = lambda b: "".join("0" if c == "1" else "1" for c in b)   # Pulser |r> = index 0
        exact = {flip(format(i, f"0{N}b")): abs(x) ** 2 for i, x in enumerate(amp)}
        sig = max(abs(counts.get(k, 0) / tot - p) / np.sqrt(max(p * (1 - p), 1 / tot) / tot) for k, p in exact.items())
        pz = counts.get(Z2, 0) / tot
        n_cloud = np.array([sum(c * (k[i] == "1") for k, c in counts.items()) / tot for i in range(N)])
        print(f"EMU_FREE: P({Z2}) = {pz:.3f} (exact {exact[Z2]:.3f}) from {tot} shots; "
              f"largest deviation {sig:.1f} sigma; max|Δn| {np.abs(n_cloud - n_own).max():.3f}")
        print("PASS" if sig < 4.5 else "FAIL")
        fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.6]})
        suptitle(fig, f"Pasqal Cloud EMU_FREE: 9-atom Z2 sweep, {tot} shots vs exact")
        ax = axs[0]; sites = np.arange(N)
        exact_line(ax, sites, n_own, label="exact ⟨n_i⟩ (QuTiP)")
        ax.plot(sites, n_cloud, "s", ms=10, mfc="none", mec=AQUA, mew=2, label="Pasqal Cloud shots")
        ax.errorbar(sites, n_cloud, yerr=np.sqrt(n_cloud * (1 - n_cloud) / tot), fmt="none", ecolor=AQUA, capsize=3)
        ax.set_xlabel("atom i"); ax.set_ylabel("Rydberg density ⟨n_i⟩"); ax.set_ylim(-0.05, 1.2)
        ax.set_title("Density profile"); ax.legend(loc="upper center", ncol=2, fontsize=9)
        ax = axs[1]
        keys = sorted(exact, key=lambda k: -exact[k])[:12]
        x = np.arange(len(keys))
        ax.bar(x, [counts.get(k, 0) / tot for k in keys], 0.7, color=AQUA, label="Pasqal Cloud EMU_FREE")
        ax.plot(x, [exact[k] for k in keys], "_", color=INK, ms=16, mew=2.5, label="exact probability")
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=40, ha="right", rotation_mode="anchor", fontsize=8.5, family="monospace")
        ax.set_ylabel("probability"); ax.set_title(f"12 most likely bitstrings (1 = Rydberg); largest deviation {sig:.1f}σ")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(HERE / "pasqal_rydberg_cloud.png", dpi=110)
        print("saved pasqal_rydberg_cloud.png")
        return

    # 1) Pulser emulator (ideal)
    res = pulser_run(seq)
    nt, tlist = density_pulser(res, N)
    rng = np.random.default_rng(1)
    ideal_counts = res.sample_final_state(args.shots)
    psi_p = res.get_final_state()
    # 2) own QuTiP
    n_own, psi_own, a, d = own_qutip(seq, pos)
    fid = min(1.0, abs(psi_own.overlap(qt.Qobj(psi_p.full(), dims=psi_own.dims))) ** 2)
    # 3) Qiskit Trotter
    t0 = time.time()
    n_qk, p_qk, steps, cx1 = qiskit_trotter(pos, a, d)
    t_qk = time.time() - t0
    err_p = np.abs(nt[-1] - n_own).max(); err_q = np.abs(n_qk - n_own).max()
    pz_exact = abs(psi_own.full().ravel()[int("".join("0" if c == "1" else "1" for c in Z2), 2)]) ** 2
    pz_qk = p_qk.get(Z2[::-1], 0.0)
    print(f"Pulser vs own QuTiP: fidelity {fid:.5f}, max|Δn| {err_p:.2e}")
    print(f"Qiskit Trotter ({steps} steps, dt {DT_TROT} µs) vs QuTiP: max|Δn| {err_q:.2e}  ({t_qk:.1f} s)")
    print(f"P(Z2) exact {pz_exact:.3f}, Qiskit {pz_qk:.3f}")
    ok = fid > 0.99 and err_p < 1e-2 and err_q < 1e-2
    print("PASS" if ok else "FAIL")

    # 4) noisy runs + sweep-time scan
    t0 = time.time()
    noisy = {}
    for T in T_NOISY:
        rn = pulser_run(sweep(reg, T), noise=NOISE, sampling=0.1)
        noisy[T] = rn.sample_final_state(args.shots)
    print(f"noisy runs {time.time() - t0:.0f} s")
    pz_scan = []
    for T in T_SCAN:
        st = pulser_run(sweep(reg, T)).get_final_state().full().ravel()
        pz_scan.append(abs(st[int("".join("0" if c == "1" else "1" for c in Z2), 2)]) ** 2)
    pz_noisy = [noisy[T].get(Z2, 0) / sum(noisy[T].values()) for T in T_NOISY]
    for T, p in zip(T_NOISY, pz_noisy):
        print(f"  T = {T / 1000:.0f} µs: P(Z2) ideal {pz_scan[T_SCAN.index(T)]:.3f}, noisy {p:.3f}")

    # 5) phase diagram
    t0 = time.time()
    deltas, rbs, SPi, S3 = gs_phase_diagram(N)
    print(f"phase diagram {time.time() - t0:.0f} s")

    # 6) correlations from shots
    def corr(counts):
        b = np.array([[int(c) for c in k] for k in counts]); w = np.array(list(counts.values()), float); w /= w.sum()
        n1 = w @ b; n2 = np.einsum("s,si,sj->ij", w, b, b); C = n2 - np.outer(n1, n1)
        return np.array([np.mean([C[i, i + r] for i in range(N - r)]) for r in range(N)])
    g_id, g_no = corr(ideal_counts), corr(noisy[T_NS])

    # ------------------------------------------------------------ figure
    fig, axs = plt.subplots(3, 3, figsize=(16, 14.5))
    suptitle(fig, "Pasqal neutral atoms (Pulser): adiabatic sweep into the Z2 Rydberg crystal, 9 atoms")

    ax = axs[0, 0]
    tr = np.array(list(lay.traps_dict.values()))
    ax.scatter(tr[:, 0], tr[:, 1], s=40, facecolors="none", edgecolors=GRID, lw=1.2, label="empty trap (pre-calibrated layout)")
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=n_own, cmap="Blues", vmin=0, vmax=1, s=220, edgecolors=INK, lw=1.2, zorder=3)
    for i in range(0, N, 2):
        ax.add_patch(plt.Circle(pos[i], Rb / 2, fill=False, ls="--", color=MAGENTA, lw=1.2))
    ax.plot([], [], "--", color=MAGENTA, label=f"blockade radius/2 = {Rb / 2:.1f} µm")
    ax.set_aspect("equal"); ax.set_xlim(-26, 26); ax.set_ylim(-24, 30)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title("Register on AnalogDevice traps\n(colour = final Rydberg density)")
    ax.legend(loc="upper center", fontsize=8.5, ncol=1)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02); cb.set_label("⟨n_i⟩ final")

    ax = axs[0, 1]
    ts = np.arange(len(a)) * 1e-3
    ax.plot(ts, a / (2 * np.pi), color=BLUE, lw=2, label="Ω(t) / 2π  (Rabi)")
    ax.plot(ts, d / (2 * np.pi), color=ORANGE, lw=2, label="δ(t) / 2π  (detuning)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("t (µs)"); ax.set_ylabel("MHz")
    ax.set_title("Global laser pulse = the whole program")
    ax.legend(loc="lower right", fontsize=9)

    ax = axs[0, 2]
    im = ax.imshow(nt.T, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1,
                   extent=[tlist[0], tlist[-1], -0.5, N - 0.5])
    ax.set_xlabel("t (µs)"); ax.set_ylabel("atom i")
    ax.set_title("Rydberg density ⟨n_i(t)⟩ (Pulser emulator)")
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("⟨n_i⟩")

    ax = axs[1, 0]
    sites = np.arange(N)
    exact_line(ax, sites, n_own, label="own QuTiP (exact)")
    ax.plot(sites, nt[-1], "s", ms=11, mfc="none", mec=AQUA, mew=2, label="Pulser emulator", zorder=2)
    qk_marks(ax, sites, n_qk, label=f"Qiskit Trotter ({steps} steps)")
    ax.set_xlabel("atom i"); ax.set_ylabel("final ⟨n_i⟩"); ax.set_ylim(-0.05, 1.25)
    ax.set_title("Three independent solvers agree")
    ax.legend(loc="upper center", ncol=2, fontsize=8.5)
    ax.text(0.5, 0.06, f"fidelity Pulser·QuTiP {fid:.4f}\nmax|Δn| Pulser {err_p:.1e}, Qiskit {err_q:.1e}",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=INK)

    ax = axs[1, 1]
    tot_i, tot_n = sum(ideal_counts.values()), sum(noisy[T_NS].values())
    allk = sorted(set(ideal_counts) | set(noisy[T_NS]), key=lambda k: -(ideal_counts.get(k, 0) / tot_i + noisy[T_NS].get(k, 0) / tot_n))[:8]
    x = np.arange(len(allk)); w = 0.38
    ax.bar(x - w / 2 - 0.01, [ideal_counts.get(k, 0) / tot_i for k in allk], w, color=BLUE, label="ideal")
    ax.bar(x + w / 2 + 0.01, [noisy[T_NS].get(k, 0) / tot_n for k in allk], w, color=ORANGE, label="with hardware-like noise")
    ax.set_xticks(x); ax.set_xticklabels(allk, rotation=40, ha="right", rotation_mode="anchor", fontsize=8, family="monospace")
    ax.set_ylabel(f"probability ({args.shots} shots)")
    ax.set_title(f"Measured bitstrings, T = {T_NS / 1000:.0f} µs (1 = Rydberg)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axs[1, 2]
    ax.plot(np.array(T_SCAN) / 1000, pz_scan, "-o", color=BLUE, lw=2, ms=8, label="ideal")
    ax.plot(np.array(T_NOISY) / 1000, pz_noisy, "-s", color=ORANGE, lw=2, ms=8, label="hardware-like noise")
    ax.axvline(DEV.max_sequence_duration / 1000, color=MUTED, ls=":", lw=1.2)
    ax.text(DEV.max_sequence_duration / 1000 - 0.1, 0.02, "device max", ha="right", fontsize=8.5, color=MUTED)
    ax.set_xlabel("sweep time T (µs)"); ax.set_ylabel(f"P({Z2})")
    ax.set_title("Adiabaticity vs decoherence"); ax.set_ylim(0, 1); ax.legend(loc="upper left", fontsize=9)

    ax = axs[2, 0]
    im = ax.pcolormesh(deltas, rbs, SPi, cmap="Blues", shading="auto")
    ax.contour(deltas, rbs, S3, levels=[0.5 * S3.max()], colors=[ORANGE], linewidths=2)
    ax.plot(DF / OMEGA, Rb / a_sp, "*", ms=16, color=YELLOW, mec=INK, zorder=4)
    ax.annotate("this sweep\nends here", (DF / OMEGA, Rb / a_sp), (DF / OMEGA + 1.3, Rb / a_sp + 0.15),
                fontsize=8.5, ha="left", color=SURF, arrowprops=dict(arrowstyle="->", color=SURF))
    DD, RR = np.meshgrid(deltas, rbs)
    for S, lab, col in ((S3, "Z3", ORANGE), (SPi, "Z2", SURF)):
        m = S > 0.7 * S.max()
        ax.text(DD[m].mean(), RR[m].mean(), lab, color=col, fontsize=12, fontweight="bold", ha="center", va="center")
    ax.text(-0.6, 2.4, "disordered", color=INK, fontsize=9.5, ha="left")
    ax.plot([], [], color=ORANGE, lw=2, label="Z3 order: S(2π/3) = ½ max")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("δ / Ω"); ax.set_ylabel("R_b / a")
    ax.set_title("Ground-state phase diagram (exact, 9 atoms)")
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("Z2 structure factor S(π)")

    ax = axs[2, 1]
    rr = np.arange(N)
    ax.plot(rr, g_id, "-o", color=BLUE, lw=2, ms=8, label="ideal")
    ax.plot(rr, g_no, "-s", color=ORANGE, lw=2, ms=8, label="hardware-like noise")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("distance r (sites)"); ax.set_ylabel("C(r) = ⟨n_i n_i+r⟩ − ⟨n_i⟩⟨n_i+r⟩")
    ax.set_title("Antiferromagnetic correlations from shots"); ax.legend(loc="upper right", fontsize=9)

    ax = axs[2, 2]
    Ns = np.arange(4, 81)
    ax.semilogy(Ns, 2 * (Ns - 1) * steps, color=BLUE, lw=2, label="gate model, nearest-neighbour only")
    ax.semilogy(Ns, Ns * (Ns - 1) * steps, color=VIOLET, lw=2, label="gate model, all C6/r⁶ pairs")
    ax.semilogy([N], [cx1 * steps], "o", ms=9, mfc="none", mec=QK, mew=2, label=f"Qiskit transpiled, N = {N}")
    ax.axvline(DEV.max_atom_num, color=MUTED, ls=":", lw=1.2)
    ax.text(DEV.max_atom_num - 1, 4e6, f"AnalogDevice: max {DEV.max_atom_num} atoms,\nalways one {T_NS / 1000:.0f} µs pulse", ha="right", va="top", fontsize=8.5, color=MUTED)
    ax.set_xlabel("number of atoms N"); ax.set_ylabel(f"CX gates to emulate one {T_NS / 1000:.0f} µs pulse")
    ax.set_title(f"Digital cost of one analog pulse ({steps} Trotter steps)")
    ax.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = HERE / "pasqal_rydberg.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out.name}  (total {time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
