from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite


@dataclass(frozen=True)
class SparkXResult:
    table: pd.DataFrame

    def top_genes(
        self,
        *,
        n: int = 1000,
        sort_by: Literal["adjustedPval", "combinedPval"] = "adjustedPval",
    ) -> list[str]:
        df = self.table
        key = sort_by if sort_by in df.columns else "combinedPval"
        if key not in df.columns:
            raise ValueError("SPARK-X result missing p-value columns.")
        return df.sort_values(key, ascending=True)["gene"].head(int(n)).astype(str).tolist()


def _default_runner_path() -> Path:
    return Path(__file__).with_name("_sparkx_runner.R")


def run_sparkx(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
    rscript: str = "Rscript",
    num_cores: int = 1,
    option: str = "mixture",
    exclude_mt: bool = True,
    mt_prefixes: tuple[str, ...] = ("mt-", "MT-"),
    min_spot_total_counts: int = 1,
    min_gene_nonzero_spots: int = 5,
    min_gene_total_counts: int = 10,
    max_genes: int | None = None,
    output_dir: str | Path | None = None,
    runner_r_path: str | Path | None = None,
) -> SparkXResult:
    """Run SPARK-X (R package SPARK) on an AnnData with raw counts in `adata.X`.

    - Assumes `adata.X` is **raw counts** (as you confirmed).
    - Coordinates are read from `adata.obsm[spatial_key]` (shape n_obs x 2).

    Returns a DataFrame containing SPARK-X `res_mtest` with BH-adjusted p-values.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"Missing `adata.obsm['{spatial_key}']`.")
    xy = np.asarray(adata.obsm[spatial_key], float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"`adata.obsm['{spatial_key}']` must be shape (n_obs, 2).")

    X = adata.X
    if sp.issparse(X):
        X_sp = X.tocsr()
    else:
        X_sp = sp.csr_matrix(np.asarray(X))
    genes = adata.var_names.astype(str).to_numpy()
    barcodes = adata.obs_names.astype(str).to_numpy()

    # Filter spots (library size)
    spot_sum = np.asarray(X_sp.sum(axis=1)).ravel()
    keep_spot = spot_sum >= int(min_spot_total_counts)
    if not np.all(keep_spot):
        X_sp = X_sp[keep_spot, :]
        xy = xy[keep_spot, :]
        barcodes = barcodes[keep_spot]

    # Filter genes by sparsity / total counts
    gene_sum = np.asarray(X_sp.sum(axis=0)).ravel()
    gene_nz = np.asarray((X_sp > 0).sum(axis=0)).ravel()
    keep_gene = (gene_nz >= int(min_gene_nonzero_spots)) & (gene_sum >= int(min_gene_total_counts))
    if exclude_mt:
        for pfx in mt_prefixes:
            keep_gene &= ~np.char.startswith(genes.astype(str), pfx)

    if not np.all(keep_gene):
        keep_idx = np.where(keep_gene)[0]
        X_sp = X_sp[:, keep_idx]
        genes = genes[keep_idx]
        gene_sum = gene_sum[keep_idx]

    # Optional cap for speed: keep top genes by total counts after filtering.
    if max_genes is not None:
        mg = int(max_genes)
        if mg > 0 and genes.shape[0] > mg:
            top = np.argsort(-gene_sum)[:mg]
            X_sp = X_sp[:, top]
            genes = genes[top]

    # AnnData is (spots x genes). SPARK expects (genes x spots).
    counts = X_sp.T.tocoo()

    if output_dir is None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="spatiocube_sparkx_") as td:
            return run_sparkx(
                adata,
                spatial_key=spatial_key,
                rscript=rscript,
                num_cores=num_cores,
                option=option,
                exclude_mt=exclude_mt,
                mt_prefixes=mt_prefixes,
                min_spot_total_counts=min_spot_total_counts,
                min_gene_nonzero_spots=min_gene_nonzero_spots,
                min_gene_total_counts=min_gene_total_counts,
                max_genes=max_genes,
                output_dir=td,
                runner_r_path=runner_r_path,
            )

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    counts_path = outdir / "counts.mtx"
    genes_path = outdir / "genes.tsv"
    coords_path = outdir / "coords.csv"
    out_csv = outdir / "sparkx_res.csv"

    mmwrite(str(counts_path), counts)
    genes_path.write_text("\n".join(genes.tolist()) + "\n")

    coords = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}, index=barcodes)
    coords.to_csv(coords_path)

    runner = Path(runner_r_path) if runner_r_path is not None else _default_runner_path()
    if not runner.exists():
        raise FileNotFoundError(f"Missing SPARK-X runner script: {runner}")

    import subprocess
    import shutil

    rscript_path = str(rscript)
    if (Path(rscript_path).is_absolute() or "/" in rscript_path) and not Path(rscript_path).exists():
        alt = shutil.which("Rscript")
        if alt is None:
            raise FileNotFoundError(
                "Rscript not found.\n"
                f"Provided rscript path: {rscript_path}\n"
                "Also could not find `Rscript` on PATH. Please install R in the current environment "
                "(e.g. `micromamba install -n spatiocube -c conda-forge r-base`) or pass the correct "
                "path to `rscript=`."
            )
        rscript_path = alt

    cmd = [
        rscript_path,
        str(runner),
        str(counts_path),
        str(genes_path),
        str(coords_path),
        str(out_csv),
        str(int(num_cores)),
        str(option),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "SPARK-X failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    df = pd.read_csv(out_csv)
    # normalize column names to match common conventions
    if "combinedPval" in df.columns and "adjustedPval" not in df.columns:
        df["adjustedPval"] = df["combinedPval"]
    return SparkXResult(table=df)

