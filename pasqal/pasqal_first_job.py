"""Pasqal's "Run your first job" example (https://docs.pasqal.com/cloud/first-job/).

The sequence and the two jobs are exactly the ones in the Pasqal docs: a 2x2 square of
atoms 5 µm apart on DigitalAnalogDevice, one 100 ns constant global pulse with a
variable amplitude omega_max and detuning 2 rad/µs.

    job1 = {"runs": 20, "variables": {"omega_max": 6}}
    job2 = {"runs": 50, "variables": {"omega_max": 10.5}}

--mode local (default): runs on your CPU with pulser-simulation, which is the same
    QuTiP engine the cloud calls EMU_FREE. Free, no account.
--mode cloud: sends the batch to Pasqal Cloud's EMU_FREE (free Explorer offer). Needs
    an account at https://portal.pasqal.cloud and a project id. You are asked for the
    password in the terminal (hidden); it is not stored.

Outputs: pasqal_first_job.png and pasqal_first_job_<mode>.json
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from pulser import Pulse, Register, Sequence
from pulser.devices import DigitalAnalogDevice
from pulser_simulation import QutipEmulator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import *  # noqa: E402,F401

JOBS = [{"runs": 20, "variables": {"omega_max": 6}},
        {"runs": 50, "variables": {"omega_max": 10.5}}]


def docs_sequence():
    register = Register.square(2, spacing=5, prefix="q")
    sequence = Sequence(register, DigitalAnalogDevice)
    sequence.declare_channel("rydberg", "rydberg_global")
    omega_max = sequence.declare_variable("omega_max")
    sequence.add(Pulse.ConstantPulse(100, omega_max, 2, 0.0), "rydberg")
    return register, sequence


def exact_probs(sequence, om):
    """Exact outcome probabilities from the emulator's final state ('1' = Rydberg)."""
    res = QutipEmulator.from_sequence(sequence.build(omega_max=om)).run()
    psi = res.get_final_state().full().ravel()
    n = int(np.log2(len(psi)))
    # Pulser orders the basis with |r> = index 0, so flip every bit to get '1' = Rydberg
    return {format((2 ** n - 1) ^ i, f"0{n}b"): abs(a) ** 2 for i, a in enumerate(psi)}, res


def run_local(sequence):
    np.random.seed(0)
    out = []
    for job in JOBS:
        probs, res = exact_probs(sequence, job["variables"]["omega_max"])
        out.append(dict(res.sample_final_state(N_samples=job["runs"])))
    return out


def run_cloud(sequence):
    from pasqal_cloud import PasqalCloudConnection, RemoteEmuFreeBackend
    username = os.environ.get("PASQAL_USERNAME") or input("Pasqal username (e-mail): ").strip()
    project_id = os.environ.get("PASQAL_PROJECT_ID") or input("project id: ").strip()
    connection = PasqalCloudConnection(username=username, project_id=project_id)  # prompts for the password
    backend = RemoteEmuFreeBackend(sequence=sequence, connection=connection)
    results = backend.run(job_params=JOBS, wait=True)
    print("batch id:", results.batch_id)
    out = []
    for job_id, result in results.get_available_results().items():
        print(f"job-id: {job_id}, result: {dict(result.final_bitstrings)}")
        out.append(dict(result.final_bitstrings))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "cloud"], default="local")
    args = ap.parse_args()
    register, sequence = docs_sequence()
    counts = run_cloud(sequence) if args.mode == "cloud" else run_local(sequence)
    exact = [exact_probs(sequence, j["variables"]["omega_max"])[0] for j in JOBS]

    ok = True
    for job, c, p in zip(JOBS, counts, exact):
        om, runs = job["variables"]["omega_max"], job["runs"]
        top = sorted(p.items(), key=lambda kv: -kv[1])[:4]
        # few shots only allow a loose check: every outcome within 4 binomial sigma of exact
        dev = max(abs(c.get(k, 0) / runs - v) / np.sqrt(max(v * (1 - v), 1 / runs) / runs) for k, v in p.items())
        ok &= sum(c.values()) == runs and dev < 4 and abs(sum(p.values()) - 1) < 1e-6
        print(f"omega_max = {om}: {runs} runs -> {dict(sorted(c.items(), key=lambda kv: -kv[1]))}")
        print(f"   exact top: {[(k, round(float(v), 3)) for k, v in top]}   largest deviation {dev:.1f} sigma")
    print("PASS" if ok else "FAIL")
    json.dump({"mode": args.mode, "jobs": JOBS, "counts": counts, "exact": exact},
              open(HERE / f"pasqal_first_job_{args.mode}.json", "w"), indent=1)

    # ------------------------------------------------------------ figure
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.9), gridspec_kw={"width_ratios": [0.8, 1, 1.5, 1.5]})
    suptitle(fig, f"Pasqal docs 'Run your first job': 2×2 atoms, one 100 ns pulse  ({args.mode} EMU_FREE engine)")

    ax = axs[0]
    pos = np.array([np.asarray(v) for v in register.qubits.values()])
    for (x, y), name in zip(pos, register.qubits):
        ax.add_patch(plt.Circle((x, y), DigitalAnalogDevice.rydberg_blockade_radius(6) / 2, fill=False, ls="--", color=MAGENTA, lw=1))
        ax.text(x, y + 1.0, name, ha="center", fontsize=9, color=INK)
    ax.scatter(pos[:, 0], pos[:, 1], s=200, color=BLUE, edgecolors=INK, zorder=3)
    ax.plot([], [], "--", color=MAGENTA, label="blockade radius/2 (Ω = 6)")
    ax.set_aspect("equal"); ax.set_xlim(-8, 8); ax.set_ylim(-8, 10.5)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_title("Register"); ax.legend(loc="upper center", fontsize=8)

    ax = axs[1]
    t = [0, 0, 100, 100]
    for job, col in zip(JOBS, (BLUE, ORANGE)):
        om = job["variables"]["omega_max"]
        ax.plot(t, [0, om, om, 0], color=col, lw=2, label=f"Ω, job omega_max = {om}")
    ax.plot([0, 100], [2, 2], color=MUTED, lw=2, ls=":", label="detuning δ = 2")
    ax.set_xlabel("t (ns)"); ax.set_ylabel("rad/µs"); ax.set_ylim(0, 14.5)
    ax.set_title("Pulse (variable amplitude)"); ax.legend(loc="upper center", fontsize=8)

    for ax, job, c, p, col in zip(axs[2:], JOBS, counts, exact, (BLUE, ORANGE)):
        om, runs = job["variables"]["omega_max"], job["runs"]
        keys = sorted(p, key=lambda k: (k.count("1"), k))
        x = np.arange(len(keys))
        ax.bar(x, [c.get(k, 0) / runs for k in keys], 0.7, color=col, label=f"{runs} shots ({args.mode})")
        ax.plot(x, [p[k] for k in keys], "_", color=INK, ms=14, mew=2.5, label="exact probability")
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=60, fontsize=8, family="monospace")
        ax.set_ylabel("probability"); ax.set_xlabel("measured bitstring (1 = Rydberg)")
        ax.set_title(f"Job omega_max = {om}, {runs} runs"); ax.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(HERE / "pasqal_first_job.png", dpi=110)
    print("saved pasqal_first_job.png")


if __name__ == "__main__":
    main()
