from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np


ArrayLike = np.ndarray
JsonLike = Any


@dataclass(frozen=True)
class SliceInfo:
    """Lightweight metadata for a slice in a SpatioCube."""

    key: str  # e.g. sampleid
    z: float
    n_obs: int
    extra: Mapping[str, JsonLike] | None = None


ObsKey = str
ObsmKey = str
UnsDict = MutableMapping[str, Any]
SliceKey = str
SliceOrder = Sequence[str]
