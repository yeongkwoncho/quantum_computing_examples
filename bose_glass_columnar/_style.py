"""Shared figure style for the physics examples (validated categorical palette)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
QK, PL = BLUE, ORANGE  # Qiskit, PennyLane - fixed across every figure

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False,
})

def exact_line(ax, x, y, label="Exact (QuTiP)", **kw):
    return ax.plot(x, y, color=INK, lw=2, label=label, zorder=1, **kw)

def qk_marks(ax, x, y, label="Qiskit", **kw):
    return ax.plot(x, y, "o", ms=8, mfc="none", mec=QK, mew=2, label=label, zorder=3, **kw)

def pl_marks(ax, x, y, label="PennyLane", **kw):
    return ax.plot(x, y, "x", ms=7, color=PL, mew=2, label=label, zorder=4, **kw)

def suptitle(fig, text):
    fig.suptitle(text, fontsize=14, fontweight="bold", color=INK, x=0.01, ha="left")
