#!/bin/bash

python /data/home/natant/Negatives/TFBS_negatives/utils/ENCODE_chip/Create_H5_dataset.py --Gen_h5t --sampl_dinucl_matched --Gen_dinucl_shuffled --bed_location /data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC --h5t_location /data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr
#python /data/home/natant/Negatives/TFBS_negatives/utils/ENCODE_chip/Create_H5_dataset.py --sampl_dinucl_matched --h5t_location /data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5