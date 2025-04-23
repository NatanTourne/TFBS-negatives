#! VERY WEIRD: THIS SCRIPT TAKES 101BP seqs as inputs but outputs 100BP negatives!!!! WHY???
library(gkmSVM)
library(BSgenome.Hsapiens.UCSC.hg38.masked)
# input_folder <- "/data/home/natant/Negatives/testing_ground/20250404_temp/"
# output_folder <- "/data/home/natant/Negatives/testing_ground/20250404_temp/"
# bed_files <- list.files(input_folder, pattern = "\\.bed$", full.names = TRUE)
# for (input_bed in bed_files) {
#     output_bed <- file.path(output_folder, paste0(gsub("_positive_sequences$", "", tools::file_path_sans_ext(basename(input_bed))), "_negatives.bed"))
#     genNullSeqs(input_bed, genome = BSgenome.Hsapiens.UCSC.hg38.masked, outputBedFN = output_bed, nMaxTrials = 10, xfold = 1, length_match_tol=0)
# }

args <- commandArgs(trailingOnly = TRUE)


input_folder <- args[1]
xfold <- as.numeric(args[2])

if (!dir.exists(input_folder)) {
    stop("Input folder does not exist.")
}


bed_files <- list.files(input_folder, pattern = "\\.bed$", full.names = TRUE)
for (input_bed in bed_files) {
    output_bed <- file.path(input_folder, paste0(gsub("_positive_temp$", "", tools::file_path_sans_ext(basename(input_bed))), "_negatives.bed"))
    genNullSeqs(input_bed, genome = BSgenome.Hsapiens.UCSC.hg38.masked, outputBedFN = output_bed, nMaxTrials = 10, xfold = xfold, length_match_tol = 0)
}


# input_bed <- "/data/home/natant/Negatives/testing_ground/20250402_test/20250402_test_longer_ATF3_positive_sequences.bed"
# output_bed <- "/data/home/natant/Negatives/testing_ground/20250402_test/20250402_test_longer_ATF3_positive_sequences_null.bed"
# genNullSeqs(input_bed, genome = BSgenome.Hsapiens.UCSC.hg38.masked, outputBedFN = output_bed,nMaxTrials=10,xfold=1)

# genNullSeqs('/data/home/natant/Negatives/testing_ground/20250402_test/CTCF_GM12878_hg38_top5k.bed',nMaxTrials=10,xfold=1,genome=BSgenome.Hsapiens.UCSC.hg38.masked, outputPosFastaFN='/data/home/natant/Negatives/testing_ground/20250402_test/CTCF_GM12878_hg38_top5k.fa', outputBedFN='/data/home/natant/Negatives/testing_ground/20250402_test/neg1x_CTCF_GM12878_hg38_top5k.bed', outputNegFastaFN='/data/home/natant/Negatives/testing_ground/20250402_test/neg1x_CTCF_GM12878_hg38_top5k.fa')
