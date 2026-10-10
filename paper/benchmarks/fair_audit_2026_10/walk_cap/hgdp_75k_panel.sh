#!/bin/bash
# Walk cap on the 75,552-hap v3 panel, accuracy on a LARGE out-of-panel cohort: 925 HGDP individuals (not in UKB or 1KG),
# chr22, masked to the Table 4c common-SNP array (1KGP AF 1-99% SNPs, one per 5 kb), scored against their own genotypes.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; S=$R/target/release/selphi; HG=/data/tmp/hgdp; W=/data/tmp/member2/hgdp75k; cd $W
GN=s3://gnomad-public-us-east-1/resources/hgdp_1kg/phased_haplotypes_v2; c=22; bcf=chr$c.bcf
B=/data/projects/wgs_imputation_benchmark; P=$B/panel/srp/chr${c}_v3.srp
[ -s $bcf ] || { aws s3 cp --no-sign-request $GN/hgdp1kgp_chr${c}.filtered.SNV_INDEL.phased.shapeit5.bcf $bcf >/dev/null && aws s3 cp --no-sign-request $GN/hgdp1kgp_chr${c}.filtered.SNV_INDEL.phased.shapeit5.bcf.csi $bcf.csi >/dev/null; }
bcftools view -S $HG/ref_1kgp.txt -m2 -M2 --force-samples $bcf -Ou 2>/dev/null | bcftools +fill-tags -- -t AF 2>/dev/null | bcftools view -v snps -i 'INFO/AF>=0.01 && INFO/AF<=0.99' -Ou 2>/dev/null \
  | bcftools query -f '%CHROM\t%POS\n' | awk 'BEGIN{last=-1e9}{if($2-last>=5000){print $1"\t"$2; last=$2}}' > array_sites.txt
bcftools view -S $HG/hgdp_targets.txt -m2 -M2 --force-samples $bcf -Ob -o truth.bcf --threads 8 2>/dev/null && bcftools index -f truth.bcf
bcftools view -S $HG/hgdp_targets.txt -T array_sites.txt --force-samples $bcf 2>/dev/null | sed '/^#/!s/|/\//g' | bgzip -@8 > array.vcf.gz && tabix -f array.vcf.gz
echo "SETUP array $(bcftools index -n array.vcf.gz) sites, truth $(bcftools index -n truth.bcf) records, $(bcftools query -l array.vcf.gz | wc -l) samples"
cat $P > /dev/null
for K in 0 200 400 800; do
  /usr/bin/time -v env SELPHI_PBWT_WALK_CAP=$K $S --refpanel $P --input array.vcf.gz --map $B/maps/plink.chr$c.GRCh38.map \
     --truth truth.bcf --exclude-panel-monomorphic $P --out k$K --threads 16 > k$K.log 2> k$K.t
  python3 - k$K.eval.json K$K "$(awk '/Elapsed \(wall/{print $8}' k$K.t)" <<'PY'
import json,sys
j=json.load(open(sys.argv[1])); b=['0.05-0.1%','0.1-0.2%','0.2-0.5%','0.5-1%','1-2%','2-5%','5-10%','10-20%','20-50%']
print(f"RESULT HGDP75K {sys.argv[2]} wall {sys.argv[3]} overall {j['overall']['mean_r2']:.5f} ps {j['per_sample_mean_r2']:.5f} n {j['overall']['n']} | "+' '.join(('%.4f'%j[k]['mean_r2']) if j.get(k,{}).get('n') else 'NA' for k in b))
PY
  rm -f k$K.vcf.gz*
done
echo HGDP75KDONE
