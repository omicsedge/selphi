# gets path to the directory of this file
declare -rx thisdir="`dirname "$BASH_SOURCE"`"

chipsites_input="$1"
fullseq_refpan="$2"


if [[ -z "$fullseq_refpan" ]]; then
    echo -e "usage:"
    echo -e " bash \"$0\" <input chip-sites dataset.vcf.gz> <reference panel dataset.vcf.gz>"
    exit 10
fi
if [[ ! -f "$chipsites_input" ]]; then
    echo -e "ERROR: input dataset file does not exist at:"
    echo -e " • \"$chipsites_input\""
    exit 11
fi
if [[ ! -f "$fullseq_refpan" ]]; then
    echo -e "ERROR: reference panel dataset file does not exist at:"
    echo -e " • \"$fullseq_refpan\""
    exit 12
fi


echo -e "1. slicing only chip sites (input samples sites) from the reference panel ..."
time { bash "${thisdir}"/cr8-chip-sites-vcf.sh "$chipsites_input" "$fullseq_refpan" ; }

echo -e "\n2. extracting metadata 1/2 ..."
time { python3 "${thisdir}"/chip-sites-idxs-in-full-seq.py "$chipsites_input" "$fullseq_refpan" ; }

echo -e "\n3. extracting metadata 2/2 ..."
time { bash "${thisdir}"/extract_vcf_metadata.sh "$fullseq_refpan" ; }

echo -e "done"

