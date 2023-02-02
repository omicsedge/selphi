extract_vcf_metadata(){
    local vcffile="$1"

    if ! [[ -f "$vcffile" ]]; then
        echo -e "Error: no such file: \"$vcffile\""
        return 2
    fi


    file_type="$(file -b --mime-type "$(realpath "$vcffile")")"
    if [[ "$file_type" == 'application/gzip' ]] || [[ "$file_type" == 'application/x-gzip' ]]; then
        output(){
            gunzip -c "$1"
        }
    else
        output(){
            cat "$1"
        }
    fi

    local -i table_header_line="$(output "$vcffile" | grep -n -m 1 -v '^##' | cut -d: -f1)"
    local -i vcf_header_line=$((table_header_line-1))
    local -i first_data_row_num=$((table_header_line+1))

    output_vcf_header(){
        output "$vcffile" | head -n "$vcf_header_line"
    }
    output_table_header(){
        output "$vcffile" | head -n "$table_header_line" | tail -n 1
    }
    output_table(){
        output "$vcffile" | tail -n +"$first_data_row_num"
    }



    local vcfheader_file="${vcffile}.vcfheader.txt"
    if [[ -f "${vcfheader_file}" ]]; then
        echo " - \"$vcfheader_file\" already exists"
    else
        echo "creating: \"$vcfheader_file\""
        output_vcf_header > "${vcfheader_file}"
    fi

    local vcf_metadata_cols_file="${vcffile}.cols1-9.tsv"
    if [[ -f "${vcf_metadata_cols_file}" ]]; then
        echo " - \"$vcf_metadata_cols_file\" already exists"
    else
        echo "creating: \"$vcf_metadata_cols_file\""
        (output_table_header ; output_table) | cut -d$'\t' -f1-9 > "${vcf_metadata_cols_file}"
    fi

}


extract_vcf_metadata "$1"
# extract_vcf_metadata "data/separated_datasets/chr20/reference_panel.30x.hg38_chr20_noinfo.fullseq-2910-refpan-samples.vcf.gz"

