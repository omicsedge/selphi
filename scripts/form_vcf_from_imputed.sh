#!/bin/bash

form_vcf(){

    declare vcfheader_file="$1"
    declare vcfcolumns1thru9="$2"
    declare results_vcf="$3"
    shift; shift; shift
    # the rest should be genotype columns


    # if ! [[ -f "$thefile" ]]; then
    #     echo -e "Error: no such file: \"$thefile\""
    #     return 2
    # fi

    echo -e "$(date)\tforming a vcf into \"$results_vcf\" ..."


    # echo -e "using vcf header from: \"$vcfheader_file\""
    # echo -e "joining: "
    # echo -e " • \"$vcfcolumns1thru9\""
    # for GT_col in "$@"; do
    #     echo -e " • \"$GT_col\""
    # done


    (
        cat "$vcfheader_file" ;
        paste -d$'\t'  <( cat "$vcfcolumns1thru9" )  "$@" ;
    ) | gzip > "$results_vcf"


    echo -e "$(date)\t... done"

}


form_vcf "$@"

