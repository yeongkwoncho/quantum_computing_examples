"""Maximum Independent Set (MIS) with Rydberg atoms on Pasqal: a concrete optimization run.

Problem: in a graph, pick the largest set of vertices with no two of them joined by an
edge. MIS is NP-hard in general; it models e.g. placing radio transmitters, cell
towers or warehouses so that no two interfere.

Encoding (Pichler et al. 2018; Ebadi et al., Science 2022): put one atom per vertex.
Two atoms closer than the blockade radius R_b cannot both be in |r>, so a graph whose
edges are exactly "distance < R_b" (a unit-disk graph) has its independent sets as
the allowed Rydberg patterns. Sweeping the detuning from negative to positive rewards
every extra excitation, and an adiabatic sweep ends in the MIS.

Here: 10 atoms on Pasqal's pre-calibrated triangular trap layout (5 µm spacing,
R_b = 6.5 µm, so edges = nearest neighbours). The sweep time T is a Pulser *variable*,
so one batch carries three jobs (T = 1, 2.5, 5 µs) and shows how adiabaticity raises
the success rate. Results are compared with a classical brute-force solution and with
the exact quantum probabilities.

Usage:
    python pasqal_mis.py                 # local, free, no account (pulser-simulation)
    python pasqal_mis.py --mode cloud    # Pasqal Cloud EMU_FREE (free Explorer plan)
Outputs: pasqal_mis.png (local) or pasqal_mis_cloud.png, and pasqal_mis_<mode>.json
"""
import argparse, getpass, itertools, json, os, sys
from pathlib import Path
import numpy as np
from pulser import NoiseModel, Pulse, Sequence
from pulser.devices import AnalogDevice
from pulser.waveforms import InterpolatedWaveform
from pulser_simulation import QutipEmulator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import *  # noqa: E402,F401

DEV = AnalogDevice
OMEGA = 2 * np.pi * 1.8          # rad/µs -> R_b = 6.5 µm
DELTA = 2 * np.pi * 4.0          # sweep −DELTA -> +DELTA
T_LIST = [1000, 2500, 5000]      # ns, one job each
SHOTS = 500
SEED = 39                        # picks a 10-vertex graph with a unique MIS of size 5
# --noise: illustrative hardware-like noise (not a device calibration). Dephasing and
# relaxation enter the Lindblad equation; p_false_pos / p_false_neg are measurement
# errors (ground atom read as '1' / Rydberg atom read as '0').
NOISE = dict(dephasing_rate=0.2, relaxation_rate=0.01, p_false_pos=0.01, p_false_neg=0.05)


# ------------------------------------------------------------------ problem
def build_problem():
    lay = DEV.pre_calibrated_layouts[0]
    traps = lay.traps_dict
    near = [k for k, v in traps.items() if np.linalg.norm(v) < 13]
    pick = sorted(np.random.default_rng(SEED).choice(near, 10, replace=False))
    reg = lay.define_register(*pick, qubit_ids=[f"q{i}" for i in range(len(pick))])
    pos = np.array([np.asarray(p) for p in reg.qubits.values()])
    Rb = DEV.rydberg_blockade_radius(OMEGA)
    edges = [(i, j) for i, j in itertools.combinations(range(len(pos)), 2)
             if np.linalg.norm(pos[i] - pos[j]) < Rb]
    return lay, reg, pos, edges, Rb


def make_sequence(reg):
    seq = Sequence(reg, DEV)
    seq.declare_channel("ryd", "rydberg_global")
    T = seq.declare_variable("T", dtype=int)
    amp = InterpolatedWaveform(T, [1e-9, OMEGA, OMEGA, 1e-9])
    det = InterpolatedWaveform(T, [-DELTA, -DELTA, DELTA, DELTA])
    seq.add(Pulse(amp, det, 0), "ryd")
    seq.measure("ground-rydberg")
    return seq


def brute_force_mis(n, edges):
    best, sols = 0, []
    for m in range(1 << n):
        if any((m >> i) & 1 and (m >> j) & 1 for i, j in edges):
            continue
        k = bin(m).count("1")
        if k > best:
            best, sols = k, [m]
        elif k == best:
            sols.append(m)
    # Pulser bitstring: character i = atom i, '1' = Rydberg
    to_str = lambda m: "".join(str((m >> i) & 1) for i in range(n))
    return best, [to_str(m) for m in sols]


def is_independent(b, edges):
    return not any(b[i] == "1" and b[j] == "1" for i, j in edges)


def repair(b, edges):
    """Classical post-processing: drop excitations until the set is independent
    (remove the vertex with most violated edges first)."""
    s = [c == "1" for c in b]
    while True:
        bad = [(i, j) for i, j in edges if s[i] and s[j]]
        if not bad:
            return "".join("1" if x else "0" for x in s)
        cnt = {}
        for i, j in bad:
            cnt[i] = cnt.get(i, 0) + 1; cnt[j] = cnt.get(j, 0) + 1
        s[max(cnt, key=cnt.get)] = False


def score(counts, edges, mis_size, mis_set):
    tot = sum(counts.values())
    p_mis = sum(c for b, c in counts.items() if b in mis_set) / tot
    p_ind = sum(c for b, c in counts.items() if is_independent(b, edges)) / tot
    fixed = {}
    for b, c in counts.items():
        r = repair(b, edges); fixed[r] = fixed.get(r, 0) + c
    p_mis_fix = sum(c for b, c in fixed.items() if b in mis_set) / tot
    ratio = sum(c * b.count("1") for b, c in fixed.items()) / tot / mis_size
    return dict(p_mis=p_mis, p_ind=p_ind, p_mis_fix=p_mis_fix, ratio=ratio, shots=tot)


# ------------------------------------------------------------------ backends
def exact_probs(seq, T, n, noise=False):
    """Exact outcome probabilities. With noise: final density matrix of the Lindblad
    run, then the measurement errors applied analytically bit by bit."""
    cache = HERE / "pasqal_mis_noisy_reference.json"
    if noise and cache.exists():
        d = json.load(open(cache))
        if d.get("noise") == NOISE and str(T) in d["probs"]:
            return d["probs"][str(T)]
    kw = {"noise_model": NoiseModel(**NOISE)} if noise else {}
    res = QutipEmulator.from_sequence(seq.build(T=T), sampling_rate=0.25, **kw).run()
    st = res.get_final_state().full()
    diag = np.real(np.diag(st)) if noise else np.abs(st.ravel()) ** 2
    # basis index bits: atom 0 is the most significant, |r> = 0 in Pulser -> flip all bits
    P = diag[::-1].copy()
    if noise:
        fp, fn = NOISE["p_false_pos"], NOISE["p_false_neg"]
        for i in range(n):
            P = np.moveaxis(P.reshape([2] * n), i, 0)
            a0, a1 = P[0].copy(), P[1].copy()
            P[0], P[1] = a0 * (1 - fp) + a1 * fn, a0 * fp + a1 * (1 - fn)
            P = np.moveaxis(P, 0, i).ravel()
    out = {format(idx, f"0{n}b"): float(v) for idx, v in enumerate(P)}
    if noise:
        d = json.load(open(cache)) if cache.exists() else {"noise": NOISE, "probs": {}}
        if d.get("noise") != NOISE:
            d = {"noise": NOISE, "probs": {}}
        d["probs"][str(T)] = out
        json.dump(d, open(cache, "w"))
    return out


def run_local(exact, shots):
    rng = np.random.default_rng(0)
    out = []
    for p in exact:
        keys = list(p); w = np.clip(np.array([p[k] for k in keys]), 0, None)
        draw = rng.choice(len(keys), size=shots, p=w / w.sum())
        u, c = np.unique(draw, return_counts=True)
        out.append({keys[i]: int(k) for i, k in zip(u, c)})
    return out


def run_cloud(seq, shots, noise=False):
    from pasqal_cloud import PasqalCloudConnection, RemoteEmuFreeBackend
    user = os.environ.get("PASQAL_USERNAME") or input("Pasqal username (e-mail): ").strip()
    proj = os.environ.get("PASQAL_PROJECT_ID") or input("project id: ").strip()
    pw = getpass.getpass("password (hidden): ")
    conn = PasqalCloudConnection(username=user, password=pw, project_id=proj)
    del pw
    config = RemoteEmuFreeBackend.default_config
    if noise:
        config = config.with_changes(noise_model=NoiseModel(**NOISE))
    backend = RemoteEmuFreeBackend(sequence=seq, connection=conn, config=config)
    jobs = [{"runs": shots, "variables": {"T": T}} for T in T_LIST]
    print(f"submitting one batch with {len(jobs)} jobs (T = {T_LIST} ns) to EMU_FREE"
          f"{' with noise ' + str(NOISE) if noise else ''} ...")
    rr = backend.run(job_params=jobs, wait=True)
    print("batch id:", rr.batch_id)
    return [dict(r.final_bitstrings) for r in rr.results], rr.batch_id


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "cloud"], default="local")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--noise", action="store_true", help="hardware-like noise model (slow: ~15 min)")
    args = ap.parse_args()
    tag = "_noisy" if args.noise else ""
    lay, reg, pos, edges, Rb = build_problem()
    n = len(pos)
    seq = make_sequence(reg)
    mis_size, mis_list = brute_force_mis(n, edges)
    mis_set = set(mis_list)
    print(f"{n} atoms, {len(edges)} edges (R_b = {Rb:.2f} µm); classical brute force over "
          f"2^{n} = {2 ** n} subsets: MIS size {mis_size}, {len(mis_list)} optimal set(s) {mis_list}")
    batch_id = None
    if args.mode == "cloud":
        counts, batch_id = run_cloud(seq, args.shots, args.noise)
    exact = [exact_probs(seq, T, n, args.noise) for T in T_LIST]
    ideal = exact if not args.noise else [exact_probs(seq, T, n) for T in T_LIST]
    if args.mode == "local":
        counts = run_local(exact, args.shots)
    rows, ok = [], True
    for T, c, p in zip(T_LIST, counts, exact):
        s = score(c, edges, mis_size, mis_set)
        pe = sum(v for b, v in p.items() if b in mis_set)
        se = np.sqrt(max(pe * (1 - pe), 1 / s["shots"]) / s["shots"])
        dev = abs(s["p_mis"] - pe) / se
        ok &= dev < 4
        s.update(T=T, p_mis_exact=pe, dev_sigma=dev,
                 p_mis_ideal=sum(v for b, v in ideal[T_LIST.index(T)].items() if b in mis_set)); rows.append(s)
        print(f"T = {T / 1000:>3.1f} µs: P(MIS) {s['p_mis']:.3f} (exact {pe:.3f}, {dev:.1f}σ) | "
              f"independent {s['p_ind']:.2f} | after repair P(MIS) {s['p_mis_fix']:.3f}, "
              f"approx. ratio {s['ratio']:.3f}")
    print("PASS" if ok else "FAIL")
    json.dump({"mode": args.mode, "noise": NOISE if args.noise else None, "batch_id": batch_id,
               "edges": edges, "mis": mis_list, "T_ns": T_LIST, "counts": counts, "scores": rows},
              open(HERE / f"pasqal_mis{tag}_{args.mode}.json", "w"), indent=1)

    # ------------------------------------------------------------ figure
    src = "Pasqal Cloud EMU_FREE" if args.mode == "cloud" else "local pulser-simulation"
    fig, axs = plt.subplots(2, 2, figsize=(15, 11.5), gridspec_kw={"width_ratios": [1, 1.35]})
    suptitle(fig, f"Maximum Independent Set with 10 Rydberg atoms ({src}{', hardware-like noise' if args.noise else ''}, {args.shots} shots per job)")

    ax = axs[0, 0]
    tr = np.array(list(lay.traps_dict.values()))
    ax.scatter(tr[:, 0], tr[:, 1], s=30, facecolors="none", edgecolors=GRID, lw=1.1)
    for i, j in edges:
        ax.plot(*pos[[i, j]].T, color=MUTED, lw=2, zorder=1)
    best = mis_list[0]
    inm = np.array([c == "1" for c in best])
    ax.scatter(pos[~inm, 0], pos[~inm, 1], s=380, color=SURF, edgecolors=INK, lw=1.5, zorder=3, label="not in set")
    ax.scatter(pos[inm, 0], pos[inm, 1], s=380, color=BLUE, edgecolors=INK, lw=1.5, zorder=3,
               label=f"maximum independent set ({mis_size})")
    for i, (x, y) in enumerate(pos):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=9, zorder=4,
                color=SURF if inm[i] else INK, fontweight="bold")
    ax.plot([], [], color=MUTED, lw=2, label=f"edge = distance < R_b ({Rb:.1f} µm)")
    ax.set_aspect("equal"); ax.set_xlim(-17, 17); ax.set_ylim(-16, 21)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title("Graph = atoms on the trap layout")
    ax.legend(loc="upper center", fontsize=8.5, ncol=1)

    ax = axs[0, 1]
    for T, col in zip(T_LIST, (ORANGE, AQUA, BLUE)):
        s = make_sequence(reg).build(T=T)
        from pulser.sampler import sample
        ch = sample(s).channel_samples["ryd"]
        t = np.arange(len(ch.amp)) / 1000
        ax.plot(t, np.asarray(ch.det) / (2 * np.pi), color=col, lw=2, label=f"δ(t)/2π, T = {T / 1000:g} µs")
        ax.plot(t, np.asarray(ch.amp) / (2 * np.pi), color=col, lw=1.5, ls="--")
    ax.plot([], [], color=MUTED, ls="--", lw=1.5, label="Ω(t)/2π (same shape)")
    ax.set_xlabel("t (µs)"); ax.set_ylabel("MHz")
    ax.set_title("One parametrized sequence, three jobs (variable T)")
    ax.legend(loc="lower right", fontsize=8.5)

    ax = axs[1, 0]
    x = np.arange(len(T_LIST)); w = 0.26
    pm = [r["p_mis"] for r in rows]; pf = [r["p_mis_fix"] for r in rows]; pe = [r["p_mis_exact"] for r in rows]
    err = [np.sqrt(max(p * (1 - p), 1 / r["shots"]) / r["shots"]) for p, r in zip(pm, rows)]
    ax.bar(x - w, pm, w * 0.95, color=BLUE, yerr=err, capsize=3, label="raw shots: P(MIS)")
    ax.bar(x, pf, w * 0.95, color=AQUA, label="after classical repair")
    ax.bar(x + w, [r["p_ind"] for r in rows], w * 0.95, color=GRID, edgecolor=MUTED, label="shots that are independent sets")
    ax.plot(x - w, pe, "_", color=INK, ms=22, mew=2.5, label="exact P(MIS)" + (" with noise" if args.noise else ""))
    if args.noise:
        ax.plot(x - w, [r["p_mis_ideal"] for r in rows], "_", color=MAGENTA, ms=22, mew=2.5, label="exact P(MIS), no noise")
    rand = 1 / (2 ** n)
    ax.set_xticks(x); ax.set_xticklabels([f"T = {T / 1000:g} µs" for T in T_LIST])
    ax.set_ylabel("probability"); ax.set_ylim(0, 1.42); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("Success rate grows with sweep time")
    ax.legend(loc="upper center", fontsize=8.5, ncol=2)
    for xi, r in zip(x, rows):
        ax.text(xi, r["p_ind"] + 0.03, f"approx. ratio {r['ratio']:.2f}", ha="center", fontsize=8.5, color=INK)

    ax = axs[1, 1]
    c = counts[-1]; tot = sum(c.values()); p = exact[-1]
    keys = sorted(p, key=lambda k: -p[k])[:12]
    xx = np.arange(len(keys))
    cols = [BLUE if k in mis_set else (AQUA if is_independent(k, edges) else ORANGE) for k in keys]
    ax.bar(xx, [c.get(k, 0) / tot for k in keys], 0.7, color=cols)
    ax.plot(xx, [p[k] for k in keys], "_", color=INK, ms=16, mew=2.5)
    for col, lab in ((BLUE, f"MIS (size {mis_size})"), (AQUA, "independent, smaller"), (ORANGE, "violates an edge"), (INK, "exact")):
        ax.plot([], [], "s" if col != INK else "_", color=col, ms=9 if col != INK else 14, mew=2.5, label=lab)
    ax.set_xticks(xx); ax.set_xticklabels([f"{k} ({k.count('1')})" for k in keys], rotation=40, ha="right",
                                          rotation_mode="anchor", fontsize=8, family="monospace")
    ax.set_ylabel("probability"); ax.set_xlabel("bitstring, atom 0 first (set size)")
    ax.set_title(f"Most likely outcomes at T = {T_LIST[-1] / 1000:g} µs")
    ax.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    name = f"pasqal_mis{tag}.png" if args.mode == "local" else f"pasqal_mis{tag}_cloud.png"
    fig.savefig(HERE / name, dpi=110)
    print("saved", name)


if __name__ == "__main__":
    main()
