"""
_setup.py

Bootstrap partage par les 4 notebooks d'analyse exploratoire : localise la
racine du projet independamment du repertoire de travail choisi par Jupyter,
ouvre la connexion a la base et expose un helper d'enregistrement des figures.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt


def _find_project_root(marker="Lancer_application.bat"):
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Racine du projet introuvable (marqueur '{marker}').")


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "database"))
from db_connection import get_engine  # noqa: E402

# Palette et style appliques a toutes les figures, sans dependance a seaborn
# (absente du venv du projet) : une seule couleur d'accent + une grille
# discrete suffisent pour des graphs de memoire, pas besoin d'un theme complet.
COULEUR_PRINCIPALE = "#2f6690"
COULEUR_ALERTE = "#c0392b"
COULEUR_NEUTRE = "#8c96a3"
PALETTE_CATEGORIES = ["#2f6690", "#c0392b", "#3aa17e", "#e0a458", "#6c5b7b", "#43aa8b", "#a4243b"]

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

engine = get_engine()


def figure_dir(bloc):
    d = PROJECT_ROOT / "analyse_exploratoire" / "figures" / bloc
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, bloc, name):
    fig.savefig(figure_dir(bloc) / name, dpi=150, bbox_inches="tight")
