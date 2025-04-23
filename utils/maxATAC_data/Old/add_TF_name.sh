#!/bin/bash

file_dict="/data/home/natant/plmBind/Data/LINKED_DATASET/maxATAC/ChIP_Binding_File"

# TO GO TO BedGraphs
# for i in $file_dict/*.bw; do
#     [ -f "$i" ] || continue
#     echo $i
#     name=$(basename $i .bw)
#     fullpath="${file_dict}/${name}.BedGraph"
#     ./bigWigToBedGraph $i $fullpath

# done

for i in $file_dict/*.bed; do
    [ -f "$i" ] || continue
    echo $i
    name=$(basename $i .bed)
    TF=$(echo $name | cut -d "_" -f 3)
    cell=$(echo $name | cut -d "_" -f 1)
    fullpath="${file_dict}/${name}.bed"
    echo $TF
    echo $cell
    awk -i inplace -v var="$TF" '{$4=var; print}' $i 
done