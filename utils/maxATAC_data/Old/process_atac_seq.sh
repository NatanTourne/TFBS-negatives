#!/bin/bash

file_dict="/data/home/natant/plmBind/Data/LINKED_DATASET/maxATAC/ATAC_Signal_File"

#TO GO TO BedGraphs
for i in $file_dict/*.bw; do
    [ -f "$i" ] || continue
    echo $i
    name=$(basename $i .bw)
    fullpath="${file_dict}/${name}.BedGraph"
    ./bigWigToBedGraph $i $fullpath

done

for i in $file_dict/*.BedGraph; do
    [ -f "$i" ] || continue
    echo $i
    name=$(basename $i .BedGraph)
    fullpath="${file_dict}/${name}.bed"
    macs3 bdgpeakcall -i $i -o $fullpath -c 1
    #./bigWigToBedGraph $i $fullpath

done
