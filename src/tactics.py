"""Tactical style clustering and matchup features.

Each club's style is described by a slow-moving, pre-kickoff profile of its last 10
matches (built leak-free in :mod:`src.features`):

* ``ppda10``        passes allowed per defensive action; low = intense high press
* ``pass_share10``  share of passes in the pressing zone; a possession proxy (0..1)
* ``directness10``  deep completions per 100 passes; high = quick vertical attacks

K-means (k=3) on the standardised profiles yields three archetypes, named from their
centroids:

* **High-Press Possession**: the highest possession share
* **Counter / Low Block**: of the other two, the least pressing (highest PPDA)
* **Direct Transition**: the remaining cluster (presses, then attacks quickly)

The clusterer is fitted on training rows only; it never sees match outcomes, so it adds
no target leakage. Matchup features: one-hot style for each side, whether the styles
mirror each other, and two press-vs-build-up interactions.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

STYLE_NAMES = ("High-Press Possession", "Counter / Low Block", "Direct Transition")
STYLE_KEYS = ("possession", "low_block", "transition")
PROFILE = ("ppda10", "pass_share10", "directness10")

TACTIC_COLUMNS = (
    [f"h_style_{k}" for k in STYLE_KEYS] + [f"a_style_{k}" for k in STYLE_KEYS]
    + ["style_mirror", "h_press_x_a_build", "a_press_x_h_build"]
)


class TacticalStyles:
    def __init__(self, n_init: int = 20, seed: int = 42):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, n_init=n_init, random_state=seed)
        self.cluster_to_style: dict[int, int] = {}
        self.centroids: pd.DataFrame | None = None

    @staticmethod
    def _profiles(frame: pd.DataFrame, side: str) -> np.ndarray:
        return frame[[f"{side}_{c}" for c in PROFILE]].to_numpy(dtype=float)

    def fit(self, frame: pd.DataFrame) -> "TacticalStyles":
        X = np.vstack([self._profiles(frame, "h"), self._profiles(frame, "a")])
        X = X[np.isfinite(X).all(axis=1)]
        Z = self.scaler.fit_transform(X)
        self.kmeans.fit(Z)
        centres = pd.DataFrame(self.scaler.inverse_transform(self.kmeans.cluster_centers_), columns=PROFILE)
        possession = int(centres["pass_share10"].idxmax())
        rest = [i for i in range(3) if i != possession]
        low_block = max(rest, key=lambda i: centres.loc[i, "ppda10"])
        transition = next(i for i in rest if i != low_block)
        self.cluster_to_style = {possession: 0, low_block: 1, transition: 2}
        centres["style"] = [STYLE_NAMES[self.cluster_to_style[i]] for i in range(3)]
        self.centroids = centres.set_index("style").loc[list(STYLE_NAMES)]
        log.debug("Tactical centroids:\n%s", self.centroids.round(3))
        return self

    def assign(self, profiles: np.ndarray) -> np.ndarray:
        """Style index (0..2, see STYLE_NAMES) for rows of [ppda10, pass_share10, directness10]."""
        clusters = self.kmeans.predict(self.scaler.transform(np.nan_to_num(profiles)))
        return np.array([self.cluster_to_style[c] for c in clusters])

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        styles = {side: self.assign(self._profiles(frame, side)) for side in ("h", "a")}
        for side in ("h", "a"):
            for idx, key in enumerate(STYLE_KEYS):
                out[f"{side}_style_{key}"] = (styles[side] == idx).astype(float)
            out[f"{side}_style"] = [STYLE_NAMES[i] for i in styles[side]]
        out["style_mirror"] = (styles["h"] == styles["a"]).astype(float)
        # Standardised pressing (inverted PPDA) against the opponent's build-up (possession share).
        mean, scale = self.scaler.mean_, self.scaler.scale_
        press = {s: -(frame[f"{s}_ppda10"].to_numpy(float) - mean[0]) / scale[0] for s in ("h", "a")}
        build = {s: (frame[f"{s}_pass_share10"].to_numpy(float) - mean[1]) / scale[1] for s in ("h", "a")}
        out["h_press_x_a_build"] = press["h"] * build["a"]
        out["a_press_x_h_build"] = press["a"] * build["h"]
        return out

    def describe(self) -> list[dict]:
        if self.centroids is None:
            return []
        return [{"style": name, "ppda": round(float(row["ppda10"]), 2),
                 "pass_share": round(float(row["pass_share10"]), 3),
                 "directness": round(float(row["directness10"]), 2)}
                for name, row in self.centroids.iterrows()]
