<!-- dx-header -->
# Selphi genotype imputation app (DNAnexus Platform App)

Weighted-PBWT genotype imputation algorithm

This is the source code for an app that runs on the DNAnexus Platform.
For more information about how to run or modify it, see
https://documentation.dnanexus.com/.
<!-- /dx-header -->

#### Architecture

Selphi runs natively on DNAnexus instances (no Docker). The app wrapper:

1. Installs Python 3.11 from deadsnakes PPA (~1 min)
2. Installs selphi Python dependencies via pip (~2-3 min)
3. Downloads input files from DNAnexus storage
4. Runs `selphi.py` directly via `python3.11`
5. Uploads output files back to DNAnexus storage

Selphi source code is bundled into the app at build time via a Makefile that copies it into `resources/`. Total setup time is ~3-4 minutes; imputation jobs typically run 5-60+ minutes, so this overhead is minimal.

#### Building the app

```bash
cd apps/selphi-imputation
dx build -a .
```

The `Makefile` runs automatically during `dx build`, copying selphi source code into `resources/home/dnanexus/selphi/`. No Docker image is needed.

#### Running the app
1. Make sure your input data is uploaded on the platform
2. Prepare a batch file using the dx tool kit. Refer to the [Docs](https://documentation.dnanexus.com/user/running-apps-and-workflows/running-batch-jobs).

**Generate Batch Inputs**

```bash
dx generate_batch_inputs \
	-itarget='target-(.*).bcf.gz' \
	--path='/adriano/target'
```
This command generates batch inputs. The input files are specified with a regular expression `-itarget='target-(.*).bcf.gz'`, and the input files are located in the specified path `--path='/adriano/target'`.

| batch ID | target |	target ID |
|----------|--------|-----------|
|0         | target0.vcf.gz |	project-ID:file-ID |
|1         | target1.vcf.gz |	project-ID:file-ID |
|2         | target2.vcf.gz |	project-ID:file-ID |

**Prepare reference panel for Selphi imputation**
```bash
# From VCF/BCF source (most common)
for CHR in {1..22}; do
    file_id=$(dx find data --name chr${CHR}_ref.vcf.gz --brief)
    file_id=${file_id#*:}
    dx run selphi-imputation \
        -iprepare_reference=True \
        -iref_source_vcf="$file_id" \
        -irefpanel="/reference/chr${CHR}" \
        -icores=12 \
        --instance-type="mem3_ssd2_v2_x16" \
        --priority low \
        --name "selphi chr${CHR} ref preparation" \
        --yes
done
```

```bash
# From XSI source (requires xSqueezeIt compilation, adds ~1 min to setup)
for CHR in {1..22}; do
    file_id=$(dx find data --name chr${CHR}_ukb_100k.xsi --brief)
    file_id=${file_id#*:}
    dx run selphi-imputation \
        -iprepare_reference=True \
        -iref_source_xsi="$file_id" \
        -irefpanel="/reference/chr${CHR}_ukb_100k" \
        -icores=12 \
        --instance-type="mem3_ssd2_v2_x16" \
        --priority low \
        --name "selphi chr${CHR} ref preparation" \
        --yes
done
```

**Run Selphi imputation**
```bash
dx run selphi-imputation \
   -itarget='file-id-of-target-vcf' \
   -irefpanel='/path/to/reference/prefix-name-no-extension' \
   -imap='file-id-of-map-file-to-use' \
   -icores=10 \
   -ioutvcf='/path/to/output/prefix-name-no-extension' \
   --instance-type='mem2_ssd1_v2_x16' \
   --priority low \
   --name "selphi-imputation"
```

**Optional parameters**

| Parameter | Description |
|-----------|-------------|
| `match_length` | Minimum PBWT match length (default: 5) |
| `est_ne` | Estimated effective population size (default: 1000000) |
| `chunk_size` | Chunk size for SRP creation (default: 10000) |
| `no_core_reduction` | Disable automatic core reduction to limit HMM memory (default: false) |
| `tmp_path` | Location for temporary files |

**xSqueezeIt note**

The `ref_source_xsi` input requires xSqueezeIt, which is compiled from source at runtime when this input is provided (~1 min extra). Most UKBB RAP users provide VCF/BCF sources via `ref_source_vcf`, which does not require xSqueezeIt.

**Important notes for COST**

Make sure to choose the correct DNAnexus instance in dxapp.json and choose the correct number of cores. Run selphi in spot mode for optimal cost efficiency. Smaller target files work better with smaller instances.
