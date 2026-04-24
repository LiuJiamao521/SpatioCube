from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from .core import SpatioCube


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Contrastive embedding requires PyTorch. Install with `pip install torch`."
        ) from e


@dataclass(frozen=True)
class ContrastiveConfig:
    svd_dim: int = 50
    embed_dim: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.2
    epochs: int = 50
    batch_size: int = 4096
    random_state: int = 0


def _stack_X(cube: SpatioCube):
    from scipy.sparse import vstack as sp_vstack

    Xs = [a.X for a in cube.adatas]
    try:
        return sp_vstack(Xs)
    except Exception:
        return np.vstack([np.asarray(x) for x in Xs])


def contrastive_embed_3d(
    cube: SpatioCube,
    adjacency: sp.csr_matrix,
    *,
    config: ContrastiveConfig = ContrastiveConfig(),
    obsm_key: str = "X_spatiocube",
) -> np.ndarray:
    """Learn a lightweight contrastive embedding on the 3D graph.

    This is intentionally simple (no torch-geometric):
    - Node features: TruncatedSVD(X) on stacked expression
    - Encoder: 2-layer MLP
    - Positives: sample one neighbor from adjacency
    - Negatives: other nodes in batch (InfoNCE)
    """

    torch = _require_torch()
    rng = np.random.default_rng(config.random_state)

    A = adjacency.tocsr()
    A.sort_indices()
    n = A.shape[0]

    X = _stack_X(cube)
    svd = TruncatedSVD(n_components=min(config.svd_dim, X.shape[1] - 1), random_state=config.random_state)
    F = svd.fit_transform(X).astype(np.float32, copy=False)  # (n, d)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat = torch.from_numpy(F).to(device)

    enc = torch.nn.Sequential(
        torch.nn.Linear(F.shape[1], 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, config.embed_dim),
    ).to(device)

    opt = torch.optim.AdamW(enc.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    indptr = A.indptr
    indices = A.indices

    def sample_pos(i: int) -> int:
        start, end = int(indptr[i]), int(indptr[i + 1])
        if end <= start:
            return i
        return int(indices[rng.integers(start, end)])

    for _ in range(config.epochs):
        batch = rng.choice(n, size=min(config.batch_size, n), replace=False)
        pos = np.fromiter((sample_pos(int(i)) for i in batch), dtype=int, count=batch.shape[0])

        z = enc(feat[torch.from_numpy(batch).to(device)])  # (B, k)
        zp = enc(feat[torch.from_numpy(pos).to(device)])  # (B, k)

        z = torch.nn.functional.normalize(z, dim=1)
        zp = torch.nn.functional.normalize(zp, dim=1)

        logits = (z @ zp.T) / float(config.temperature)  # (B, B)
        labels = torch.arange(logits.shape[0], device=device)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        Z = enc(feat)
        Z = torch.nn.functional.normalize(Z, dim=1).cpu().numpy()

    # Write back into each slice
    offset = 0
    for a in cube.adatas:
        n_i = int(a.n_obs)
        a.obsm[obsm_key] = Z[offset : offset + n_i]
        offset += n_i

    for a in cube.adatas:
        a.uns.setdefault("SpatioCube", {}).update(
            {
                "contrastive_embed": {
                    "obsm_key": obsm_key,
                    "svd_dim": config.svd_dim,
                    "embed_dim": config.embed_dim,
                    "temperature": config.temperature,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "random_state": config.random_state,
                }
            }
        )

    return Z

