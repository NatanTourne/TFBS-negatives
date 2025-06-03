#!/bin/bash

file_dict="/data/home/natant/Negatives/Data/maxATAC/ChIP_Peaks"
file_dict_out="/data/home/natant/Negatives/Data/maxATAC/ChIP_Peaks_BedGraphs"


TO GO TO BedGraphs
for i in $file_dict/*.bw; do
    [ -f "$i" ] || continue
    echo $i
    name=$(basename $i .bw)
    fullpath="${file_dict_out}/${name}.BedGraph"
    /data/home/natant/Negatives/TFBS-negatives/utils/maxATAC_data/Old/bigWigToBedGraph $i $fullpath

done