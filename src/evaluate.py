"""Probabilistic scoring, calibration plots and confusion matrices.

Outcome order everywhere is (Home, Draw, Away), i.e. class indices 0, 1, 2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

from .features import CLASSES

log = logging.getLogger(__name__)

CLASS_NAMES = ("Home win", "Draw", "Away win")
EPS = 1e-15


def _clip(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def multiclass_brier(y: np.ndarray, probs: np.ndarray) -> float:
    """Mean over matches of sum_k (p_k - o_k)^2. 0 is perfect, 2 is worst."""
    onehot = np.eye(len(CLASSES))[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((np.asarray(probs) - onehot) ** 2, axis=1)))


def ranked_probability_score(y: np.ndarray, probs: np.ndarray) -> float:
    """RPS respects the H > D > A ordering (a draw is 'closer' to a home win than an away win is)."""
    onehot = np.eye(len(CLASSES))[np.asarray(y, dtype=int)]
    cum_p = np.cumsum(probs, axis=1)[:, :-1]
    cum_o = np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (len(CLASSES) - 1)))


def score(y, probs) -> dict:
    y = np.asarray(y, dtype=int)
    probs = _clip(probs)
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "brier": multiclass_brier(y, probs),
        "rps": ranked_probability_score(y, probs),
        "accuracy": float(accuracy_score(y, probs.argmax(axis=1))),
    }


def base_rate_probs(y_train, n: int) -> np.ndarray:
    freq = np.bincount(np.asarray(y_train, dtype=int), minlength=3) / len(y_train)
    return np.tile(freq, (n, 1))


def bookmaker_probs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Implied probabilities from decimal odds with the overround removed proportionally."""
    odds = frame[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    valid = np.all(np.isfinite(odds) & (odds > 1.0), axis=1)
    implied = np.where(valid[:, None], 1.0 / np.where(odds > 0, odds, np.nan), np.nan)
    probs = implied / implied.sum(axis=1, keepdims=True)
    return probs, valid


def confusion(y, probs) -> np.ndarray:
    return confusion_matrix(np.asarray(y, dtype=int), np.asarray(probs).argmax(axis=1), labels=[0, 1, 2])


def plot_calibration(y, model_probs: dict[str, np.ndarray], path: Path, title: str, n_bins: int = 10) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping calibration plot")
        return None
    y = np.asarray(y, dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for k, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="Perfect")
        for name, probs in model_probs.items():
            mask = np.all(np.isfinite(probs), axis=1)
            if mask.sum() < n_bins * 5:
                continue
            frac, mean_pred = calibration_curve((y[mask] == k).astype(int), probs[mask, k],
                                                n_bins=n_bins, strategy="quantile")
            ax.plot(mean_pred, frac, marker="o", markersize=4, linewidth=1.5, label=name)
        ax.set_title(CLASS_NAMES[k])
        ax.set_xlabel("Predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Observed frequency")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_confusion(y, probs, path: Path, title: str) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping confusion matrix plot")
        return None
    cm = confusion(y, probs)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), [f"Pred {c}" for c in CLASS_NAMES])
    ax.set_yticks(range(3), [f"True {c}" for c in CLASS_NAMES])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def poisson_deviance(y, mu) -> float:
    """Mean Poisson deviance (lower is better); y*log(y/mu) is 0 when y == 0."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2 * (term - (y - mu))))


def binary_score(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
        "accuracy": round(float(np.mean((p > 0.5) == y)), 4),
    }
