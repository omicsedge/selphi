# gets path to the directory of this file
declare -rx thisdir="`dirname "$BASH_SOURCE"`"

chipsites_input="$1"
fullseq_refpan="$2"


echo -e "1. slicing only chip sites (input samples sites) from the reference panel ..."
time { bash "${thisdir}"/cr8-chip-sites-vcf.sh "$chipsites_input" "$fullseq_refpan" ; }

echo -e "\n2. extracting metadata 1/2 ..."
time { python3 "${thisdir}"/chip-sites-idxs-in-full-seq.py "$chipsites_input" "$fullseq_refpan" ; }

echo -e "\n3. extracting metadata 2/2 ..."
time { bash "${thisdir}"/extract_vcf_metadata.sh "$fullseq_refpan" ; }

echo -e "done"


# dir = f'./data/separated_datasets/chr20'
# chipsites_input = f'data/separated_datasets/chr20/reference_panel.30x.hg38_chr20_noinfo.chipsites-292-input-samples.vcf.gz'
# full_refpanel_vcfgz_path = f'data/separated_datasets/chr20/reference_panel.30x.hg38_chr20_noinfo.fullseq-2910-refpan-samples.vcf.gz'


