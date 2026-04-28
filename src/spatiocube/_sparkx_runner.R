args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript _sparkx_runner.R <counts_mtx> <genes_tsv> <coords_csv> <out_csv> [numCores] [option]")
}

counts_mtx <- args[[1]]
genes_tsv  <- args[[2]]
coords_csv <- args[[3]]
out_csv    <- args[[4]]
numCores   <- ifelse(length(args) >= 5, as.integer(args[[5]]), 1L)
option     <- ifelse(length(args) >= 6, as.character(args[[6]]), "mixture")

suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(SPARK))

cat("SPARK-X runner: reading inputs...\n", file=stderr())

sp_count <- readMM(counts_mtx)
genes <- readLines(genes_tsv)

coords <- read.csv(coords_csv, row.names = 1, check.names = FALSE)
if (!all(c("x", "y") %in% colnames(coords))) {
  stop("coords_csv must contain columns: x, y")
}
location <- as.matrix(coords[, c("x", "y")])

cat(sprintf("SPARK-X runner: matrix genes=%d spots=%d\n", nrow(sp_count), ncol(sp_count)), file=stderr())

if (nrow(sp_count) != length(genes)) {
  stop("Row count of matrix does not match genes_tsv length.")
}
if (ncol(sp_count) != nrow(location)) {
  stop("Column count of matrix does not match number of coordinates.")
}

rownames(sp_count) <- genes
colnames(sp_count) <- rownames(coords)
rownames(location) <- rownames(coords)

res <- sparkx(sp_count, location, numCores = numCores, option = option, verbose = FALSE)

cat("SPARK-X runner: finished sparkx, writing results...\n", file=stderr())

df <- as.data.frame(res$res_mtest)
df$gene <- rownames(df)

if ("combinedPval" %in% colnames(df)) {
  df$adjustedPval <- p.adjust(df$combinedPval, method = "BH")
}

df <- df[, c("gene", setdiff(colnames(df), "gene"))]
write.csv(df, out_csv, row.names = FALSE, quote = TRUE)

