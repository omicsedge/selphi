#!/bin/bash
# R7c: Table 6c (does the pedigree improve downstream imputation against an independent truth?). The June source dir was
# deleted and its Beagle arm passed `ped=`, which Beagle 5.5 does not accept. Re-run: 54 children + parents masked to GSA
# sites; phase WITHOUT pedigree (children alone) and WITH pedigree (Selphi --ped, SHAPEIT5 --pedigree; Beagle has no
# pedigree input, so its arm co-phases the parents in the cohort and is labelled as such); impute the 54 children with ONE
# fixed imputer (Selphi 2, impute-only); score imputed (non-chip) sites against each child's own WGS.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; T=data/trio_benchmark; O=/data/tmp/fair_audit/table6c; mkdir -p $O
J=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar; PC=_archive/reference_code/shapeit5/phase_common/bin/phase_common
awk '{print $2, $3, $4}' $T/trios.ped > $O/s5.ped
for C in 22 1; do
  MAP=data/maps/beagle/chr$C.map; GMAP=_archive/old_data/tests/data/genetic_maps/shapeit5_impute5/chr$C.b38.gmap.gz
  SRP=$T/ref_panel_no_trios_chr$C.srp; [ $C = 1 ] && SRP=$T/ref_panel_no_trios_chr1_v2.srp; B3=$T/ref_panel_no_trios_chr$C.bref3; BCF=$T/ref_panel_no_trios_chr$C.bcf
  if [ $C = 22 ]; then bcftools query -f '%CHROM\t%POS\n' data/target/chr22_801s_chip_unphased.vcf.gz > $O/chip$C.txt; else cp $T/chip_positions_chr1.txt $O/chip$C.txt; fi
  bcftools merge $T/children_wgs_unphased_chr$C.vcf.gz $T/fathers_wgs_chr$C.vcf.gz $T/mothers_wgs_chr$C.vcf.gz 2>/dev/null | bcftools view -T $O/chip$C.txt 2>/dev/null \
    | bcftools +setGT -- -t a -n u 2>/dev/null | bcftools view -Oz -o $O/fam$C.vcf.gz && bcftools index -f -t $O/fam$C.vcf.gz
  bcftools view -S $T/children.txt $O/fam$C.vcf.gz -Oz -o $O/kid$C.vcf.gz && bcftools index -f -t $O/kid$C.vcf.gz
  bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\n' $O/kid$C.vcf.gz > $O/chipsites$C.tsv
  echo "MASK chr$C chip sites $(bcftools index -n $O/kid$C.vcf.gz) kids $(bcftools query -l $O/kid$C.vcf.gz | wc -l) family $(bcftools query -l $O/fam$C.vcf.gz | wc -l)"
  # phasing arms -> phased CHILDREN vcf
  $S --refpanel $SRP --input $O/kid$C.vcf.gz --map $MAP --phase-only --out $O/sel_noped$C --threads 16 > $O/sel_noped$C.log 2>&1
  $S --refpanel $SRP --input $O/fam$C.vcf.gz --map $MAP --phase-only --ped $T/trios.ped --out $O/sel_ped_fam$C --threads 16 > $O/sel_ped$C.log 2>&1
  java -Xmx64g -jar $J gt=$O/kid$C.vcf.gz ref=$B3 map=$MAP out=$O/bea_noped$C impute=false nthreads=16 > $O/bea_noped$C.log 2>&1
  java -Xmx64g -jar $J gt=$O/fam$C.vcf.gz ref=$B3 map=$MAP out=$O/bea_parents_fam$C impute=false nthreads=16 > $O/bea_par$C.log 2>&1
  $PC --input $O/kid$C.vcf.gz --reference $BCF --map $GMAP --region $C --thread 16 --output $O/s5_noped$C.bcf > $O/s5_noped$C.log 2>&1
  $PC --input $O/fam$C.vcf.gz --reference $BCF --map $GMAP --region $C --pedigree $O/s5.ped --thread 16 --output $O/s5_ped_fam$C.bcf > $O/s5_ped$C.log 2>&1
  for f in sel_ped_fam bea_parents_fam s5_ped_fam; do
    src=$O/$f$C.vcf.gz; [ -s $src ] || src=$O/$f$C.bcf; tabix -f -p vcf $src 2>/dev/null; bcftools index -f $src 2>/dev/null
    bcftools view -S $T/children.txt $src -Oz -o $O/${f%_fam}$C.vcf.gz 2>/dev/null && bcftools index -f -t $O/${f%_fam}$C.vcf.gz
  done
  for f in s5_noped; do bcftools view $O/$f$C.bcf -Oz -o $O/$f$C.vcf.gz && bcftools index -f -t $O/$f$C.vcf.gz; done
  tabix -f -p vcf $O/bea_noped$C.vcf.gz 2>/dev/null
  # one fixed imputer
  for arm in sel_noped sel_ped bea_noped bea_parents s5_noped s5_ped; do
    $S --refpanel $SRP --input $O/$arm$C.vcf.gz --map $MAP --out $O/imp_$arm$C --threads 16 > $O/imp_$arm$C.log 2>&1
    $S --evaluate $O/imp_$arm$C.vcf.gz --truth $T/children_wgs_truth_chr$C.vcf.gz --exclude-sites $O/kid$C.vcf.gz --out $O/ev_$arm$C > $O/ev_$arm$C.log 2>&1
    python3 -c "import json;j=json.load(open('$O/ev_$arm$C.json'));print('RESULT T6C chr$C %-12s overall %.4f per-sample %.4f scored %d'%('$arm',j['overall']['mean_r2'],j['per_sample_mean_r2'],j['overall']['n']))"
    rm -f $O/imp_$arm$C.vcf.gz*
  done
  python3 - $O $C <<'PY'
import json,sys,numpy as np; from scipy.stats import wilcoxon
O,C=sys.argv[1],sys.argv[2]; rng=np.random.default_rng(1)
def ps(a):
    j=json.load(open(f"{O}/ev_{a}{C}.json")); p=j['per_sample']
    return {k:(v if not isinstance(v,dict) else v.get('r2',v.get('mean_r2'))) for k,v in (p.items() if isinstance(p,dict) else [(d['sample'],d['r2']) for d in p])}
for tool,(a,b) in {'selphi':('sel_noped','sel_ped'),'beagle(parents co-phased, no pedigree input)':('bea_noped','bea_parents'),'shapeit5':('s5_noped','s5_ped')}.items():
    x=ps(a); y=ps(b); k=sorted(set(x)&set(y)); d=np.array([y[s]-x[s] for s in k]); bs=[rng.choice(d,len(d)).mean() for _ in range(20000)]
    print(f"RESULT T6C-PAIRED chr{C} {tool}: +ped minus no-ped per-sample {d.mean():+.4f} CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] higher in {(d>0).sum()}/{len(d)} p {wilcoxon(d).pvalue:.2g}")
PY
  rm -f $O/*$C.vcf.gz* $O/*$C.bcf*
done
echo T6CDONE
