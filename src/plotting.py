"""Shared figure style; every figure is saved to figures/ and to report/figures/."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402


def style():
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def savefig(fig, name):
    config.FIGURES.mkdir(exist_ok=True)
    report = config.ROOT / "report" / "figures"
    report.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for folder in (config.FIGURES, report):
        fig.savefig(folder / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
