"""Self-contained publication style used by the current manuscript figures."""
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# restrained, print-safe palette (dark, saturated; not pastel, not neon)
PALETTE = {
    "line": "#111111", "accent": "#9c2b2b", "band": "#9aa7b4",
    "blue": "#1f4e79", "green": "#2e6b3e", "amber": "#b8860b",
    "grey": "#6e6e6e", "red": "#9c2b2b",
}


def set_jgr(base=13):
    """Apply JGR-style rcParams. base = base font size in pt."""
    plt.rcParams.update({
        "font.size": base,
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 1.1,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black", "ytick.color": "black",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 6.5, "ytick.major.size": 6.5,
        "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
        "axes.grid": False,
        "legend.frameon": True, "legend.edgecolor": "black",
        "legend.framealpha": 1.0, "legend.fancybox": False,
        "savefig.dpi": 300,
    })


def strip_prov(fig):
    """JGR rule: a publication figure carries NO on-figure script/provenance line. Remove any provenance
    footnote (e.g. the `save_prov` stamp `script.py | data=...` drawn at the bottom-left) if present.
    The prov JSONL log is untouched -- only the visible text goes."""
    for t in list(getattr(fig, "texts", [])):
        s = t.get_text() or ""
        try:
            y = t.get_position()[1]
        except Exception:
            y = 1.0
        if y < 0.06 and (".py" in s or s.startswith("↪")):     # bottom-strip prov stamp / "↪ src" line
            t.remove()


def finish(ax, minor_x=True, minor_y=True):
    """Show all four spines and add minor ticks. Call after plotting each Axes.
    Pass minor_x=False / minor_y=False on a log or categorical axis.
    Also strips any on-figure provenance stamp (JGR figures carry no script line)."""
    for s in ax.spines.values():
        s.set_visible(True)
    if minor_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if minor_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    if ax.figure is not None:
        strip_prov(ax.figure)
