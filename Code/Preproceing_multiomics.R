if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes")
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}
if (!requireNamespace("Seurat", quietly = TRUE)) {
    install.packages("Seurat")
}
if (!requireNamespace("Signac", quietly = TRUE)) {
    install.packages("Signac")
}
if (!requireNamespace("GenomeInfoDb", quietly = TRUE)) {
    BiocManager::install("GenomeInfoDb")
}
if (!requireNamespace("scater", quietly = TRUE)) {
    BiocManager::install("scater")
}
if (!requireNamespace("MAESTRO", quietly = TRUE)) {
    devtools::install_github("liulab-dfci/MAESTRO")
}

library(Seurat)
library(GenomeInfoDb)
library(Signac)
library(scater)
library(MAESTRO)


# h5Path = "../Data/GSM7821208_Ashkenazi_NormalBreastSample_multiome_Pool1_filtered_feature_bc_matrix.h5"
h5Path = "../Data/10k_PBMC_Multiome_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
include_tfsPath <- "../Data/TF_list.txt"
networkPath <- "../Data/network_pbmc.csv"

nfeatures <- 1000
cat("Loading data from:", h5Path, "\n")
inputdata <- Read10X_h5(h5Path)
rna_counts <- inputdata$`Gene Expression`
atac_counts <- inputdata$Peaks

cat("Preproceing data...", h5Path, "\n")
min_cell <- 0.1
min_feature <- 800


grange.counts <- StringToGRanges(rownames(atac_counts), sep = c(":", "-"))

grange.use <- seqnames(grange.counts) %in% standardChromosomes(grange.counts)

atac_counts <- atac_counts[as.vector(grange.use), ]

chrom_assay <- CreateChromatinAssay(
  counts = atac_counts,
  sep = c(":", "-"),
  min.cells = ncol(atac_counts) * min_cell
)
obj <- CreateSeuratObject(counts = chrom_assay, assay = "ATAC")


exp_assay <- CreateAssayObject(
  counts = rna_counts,
  min.cells = ncol(rna_counts) * min_cell,
  min.features = min_feature
)

obj[["RNA"]] <- exp_assay
DefaultAssay(obj) <- "RNA"

obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")

nmad <- 3
atac <- isOutlier(obj$nCount_ATAC, nmads = nmad, log = FALSE, type = "both")
rna <- isOutlier(obj$nCount_RNA, nmads = nmad, log = FALSE, type = "both")
mito <- isOutlier(obj$percent.mt, nmads = nmad, log = FALSE, type = "both")

obj <- AddMetaData(obj, atac, col.name = "atac")
obj <- AddMetaData(obj, rna, col.name = "rna")
obj <- AddMetaData(obj, mito, col.name = "mito")

obj <- subset(x = obj, subset = atac == FALSE & rna == FALSE & mito == FALSE)



DefaultAssay(obj) <- "RNA"
obj <- NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 10000)
obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = nfeatures, assay = "RNA")



pbmc_peak <- obj@assays$ATAC@counts
n <- nrow(pbmc_peak)
dia <- diag(n)
rownames(dia) <- rownames(pbmc_peak)
colnames(dia) <- 1:ncol(dia)
dia <- as(dia, "dgCMatrix")
gene_peak <- ATACCalculateGenescore(dia, organism = "GRCh38", decaydistance = 10000, model = "Enhanced")
colnames(gene_peak) <- rownames(obj@assays$ATAC@counts)

peak_count <- obj@assays$ATAC@counts
gene_count <- obj@assays$RNA@counts
peak_count[peak_count > 0] <- 1
WA <- gene_peak %*% peak_count
colnames(WA) <- colnames(peak_count)
rownames(WA) <- rownames(gene_peak)

WA <- WA[which(rowSums(as.matrix(WA)) > 0),]
WA <- WA / rowSums(WA)





if(is.null(include_tfsPath)){
  network <- read.csv(networkPath)
  tfs <- unique(network$TF)
}else{
  tfs <- read.table(include_tfsPath)
  tfs <- tfs[, 1]
}

tfs <- tfs[tfs %in% rownames(obj@assays$RNA)]
gene <- VariableFeatures(obj)
cat("Nember of TFs is:", length(tfs), "\n")



rna <- obj@assays$RNA@data
rna <- as.matrix(rna)
atac <- as.matrix(WA)

rna <- rna[intersect(union(tfs,gene), rownames(rna)), , drop = FALSE]
rna <- rna[intersect(rownames(rna), rownames(atac)), , drop = FALSE]
atac <- atac[intersect(rownames(atac), rownames(rna)), , drop = FALSE]


savePath = "../Data/"
write.table(data.frame(Gene=rownames(rna),rna),paste(savePath,"pbmc_rna.csv",sep=""),row.names=FALSE,col.names=TRUE,sep=",")
write.table(data.frame(Gene=rownames(atac),atac),paste(savePath,"pbmc_atac.csv",sep=""),row.names=FALSE,col.names=TRUE,sep=",")
cat("Finsh. ")
cat("RNA and ATAC file are saved in :", h5Path, "\n")