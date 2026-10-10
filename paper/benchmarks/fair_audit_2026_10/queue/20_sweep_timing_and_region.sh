#!/bin/bash
# R2: coverage sweep (Table 2b, Figure 4a). (a) SAME-SESSION whole-chr22 timing, HG002 1x, 16 threads:
# Selphi 2 exactly as scored (--reference -> BAQ on, current binary) vs GLIMPSE2 (GLIMPSE2_chunk + phase chunks + ligate)
# vs QUILT2 (tiled, as before). (b) accuracy parity: the competitors imputed chr22:19-31 Mb only, so Selphi is re-run
# on the same 19-31 Mb window for 3 samples x 4 depths and re-scored on the three-tool intersection.
set -uo pipefail
T=/data/tmp/lcwgs_sweep; cd $T; O=/data/tmp/fair_audit/sweep; mkdir -p $O
R=/data/projects/.claude_home/gt/selphi/mayor/rig; S=$R/target/release/selphi
G2=$R/_archive/reference_code/GLIMPSE2; QD=$R/_archive/reference_code/QUILT; export R_LIBS_USER=/data/tmp/Rlib
REF=/data/pgx/ref/GRCh38_full_analysis_set_plus_decoy_hla.fa; MAP=/data/tmp/hgdp/chr22.chr.map; bam=$T/HG002_1x.bam
sec(){ awk '/Elapsed \(wall/{n=split($8,a,":"); s=(n==3)?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]} END{printf "%.1f", s}' "$1"; }
mem(){ awk '/Maximum resident/{printf "%.1f", $6/1048576}' "$1"; }
cat panel.srp panel_poly.bcf > /dev/null
for rep in 1 2; do
  /usr/bin/time -v $S --lcwgs --refpanel panel.srp --bam $bam --reference $REF --map $MAP --out $O/sel_t$rep --threads 16 > $O/sel_t$rep.log 2> $O/sel_t$rep.time
  echo "RESULT SWEEP-TIME selphi rep$rep $(sec $O/sel_t$rep.time)s $(mem $O/sel_t$rep.time)GB load $(cut -d' ' -f1 /proc/loadavg)"; rm -f $O/sel_t$rep.vcf.gz* $O/sel_t$rep.dose.tsv.gz
done
/usr/bin/time -v $G2/chunk/bin/GLIMPSE2_chunk --input panel_poly.bcf --region chr22 --map glimpse.gmap --sequential --output $O/g_chunks.txt --threads 16 > $O/g_chunk.log 2> $O/g_chunk.time
tot=$(sec $O/g_chunk.time); mx=0; : > $O/g_glist.txt
while read idx chr irg org rest; do [ -z "$org" ] && continue
  /usr/bin/time -v $G2/phase/bin/GLIMPSE2_phase --bam-file $bam --reference panel_poly.bcf --map glimpse.gmap --input-region "$irg" --output-region "$org" --output $O/g_$idx.bcf --threads 16 > $O/g_$idx.log 2> $O/g_$idx.time
  echo $O/g_$idx.bcf >> $O/g_glist.txt; tot=$(echo "$tot + $(sec $O/g_$idx.time)" | bc); mx=$(echo "$(mem $O/g_$idx.time) $mx" | awk '{print ($1>$2)?$1:$2}')
done < $O/g_chunks.txt
/usr/bin/time -v $G2/ligate/bin/GLIMPSE2_ligate --input $O/g_glist.txt --output $O/g.bcf > $O/g_lig.log 2> $O/g_lig.time
tot=$(echo "$tot + $(sec $O/g_lig.time)" | bc); echo "RESULT SWEEP-TIME glimpse2 ${tot}s ($(wc -l < $O/g_chunks.txt) chunks) peak ${mx}GB"; rm -f $O/g_*.bcf* $O/g.bcf*
t0=$(date +%s); qmx=0
for W in 10500000 14500000 18500000 22500000 26500000 30500000 34500000 38500000 42500000 46500000 50500000; do
  RS=$W; RE=$((W+4000000-1)); qd=$O/q_$RS; rm -rf $qd; mkdir -p $qd; echo "$bam" > $qd/bl.txt
  /usr/bin/time -v Rscript $QD/QUILT2.R --outputdir=$qd --chr=chr22 --regionStart=$RS --regionEnd=$RE --buffer=500000 --bamlist=$qd/bl.txt --reference_vcf_file=$T/panel_poly.vcf.gz --genetic_map_file=$T/quilt.map --nGen=100 --nCores=16 > $qd/q.log 2> $qd/q.time
  qmx=$(echo "$(mem $qd/q.time) $qmx" | awk '{print ($1>$2)?$1:$2}'); rm -rf $qd/RData
done
echo "RESULT SWEEP-TIME quilt2 $(( $(date +%s)-t0 ))s peak ${qmx}GB"; rm -rf $O/q_*
# (b) region parity
for cov in 0.5x 1x 2x 4x; do for s in HG002 HG003 HG004; do
  $S --lcwgs --refpanel panel.srp --bam ${s}_${cov}.bam --reference $REF --map $MAP --region chr22:19000000-31000000 --out $O/reg_${s}_${cov} --threads 16 > $O/reg_${s}_${cov}.log 2>&1 || echo "RESULT REG $s $cov FAILED"
  bcftools index -t -f $O/reg_${s}_${cov}.vcf.gz 2>/dev/null
done
python3 /data/tmp/pbwt_share_2026_09_06/rare05/shared.py $cov "{\"selphi_reg\":\"$O/reg_%s_${cov}.vcf.gz\",\"selphi_whole\":\"/data/tmp/pbwt_share_2026_09_06/rare05/%s_${cov}_$( [ $cov = 0.5x ] && echo def || echo defkeep ).vcf.gz\",\"glimpse2\":\"$T/out/glimpse_%s_${cov}.bcf\",\"quilt2\":\"$T/out/quilt_%s_${cov}.vcf.gz\"}" > $O/shared_$cov.json 2> $O/shared_$cov.err
python3 - $O/shared_$cov.json $cov <<'PY'
import json,sys,statistics as st; j=json.load(open(sys.argv[1])); T=('selphi_reg','selphi_whole','glimpse2','quilt2')
ov={t:st.mean(r[t]['persample_r2'] for r in j['rows']) for t in T}; ur={t:j['pooled_bins'][t]['0-0.5']['r2'] for t in T}
print(f"RESULT REGION-PARITY {sys.argv[2]} overall "+' '.join(f"{t} {ov[t]:.4f}" for t in T)+" | ultra-rare "+' '.join(f"{t} {ur[t]:.4f}" for t in T)+f" | n {[r['n_shared'] for r in j['rows']]}")
PY
done
echo SWEEPDONE
