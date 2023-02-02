chipsites_input="$1"
fullseq_refpan="$2"




vcf_get_table(){
    local thefile="$1"

    file_type="$(file -b --mime-type "$(realpath "$thefile")")"
    if [[ "$file_type" == 'application/gzip' ]] || [[ "$file_type" == 'application/x-gzip' ]]; then
        output(){
            gunzip -c "$1"
        }
    else
        output(){
            cat "$1"
        }
    fi

    local -i table_header_line="$(output "$thefile" | grep -n -m 1 -v '^##' | cut -d: -f1)"
    local -i vcf_header_line=$((table_header_line-1))
    local -i first_data_row_num=$((table_header_line+1))

    output "$thefile" | tail -n +"$first_data_row_num"
}




variants="${chipsites_input}.chip-variants.tsv"

vcf_get_table "$chipsites_input" | cut -d$'\t' -f1-2,4-5 > "$variants"

md5sum "$variants" | cut -d " " -f1 > "${variants}.md5"
variants_md5hash="$(cat "${variants}.md5")"
variants_md5hash_16="$(cat "${variants}.md5" | cut -c -16)"

chipsites_refpan="${fullseq_refpan}.chip.${variants_md5hash}.vcf.gz"


if [[ -f "$fullseq_refpan".tbi ]]; then
    echo " - \""$fullseq_refpan".tbi\" already exists"
else
    echo "creating an index file for the full reference panel: \""$fullseq_refpan".tbi\""
    tabix "$fullseq_refpan" # around 1m
fi

if [[ -f "$chipsites_refpan" ]]; then
    echo " - \"$chipsites_refpan\" already exists"
else
    echo "filtering sites in the full reference panel by input \"chip\" sites: \"$chipsites_refpan\""
    # around 3m
    #https://www.biostars.org/p/170965/
    bcftools view -O v -R "$variants" "$fullseq_refpan" \
    | grep -Ef <(awk 'BEGIN{FS=OFS="\t";print "#"};{print "^"$1,$2,"[^\t]+",$3,$4"\t"}' "$variants") \
    > "$chipsites_refpan"
fi

