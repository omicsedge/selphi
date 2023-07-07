<!-- dx-header -->
# Selphi genotype imputation app (DNAnexus Platform App)

 weighted-PBWT genotype imputation algorithm

This is the source code for an app that runs on the DNAnexus Platform.
For more information about how to run or modify it, see
https://documentation.dnanexus.com/.
<!-- /dx-header -->

<!-- Insert a description of your app here -->

#### Building the app
Once the selphi image is created, you can save it with:
```
docker save selphi > ./apps/selphi-imputation/resources/image.tar
cd ./apps/selphi-imputation
dx build -a .
```

#### Running the app
1. Make sure your input data is uploaded on the platform
2. Prepare a batch file using the dx tool kit like. eg. (Refer to the [Docs](https://documentation.dnanexus.com/user/running-apps-and-workflows/running-batch-jobs) ) 

**Generate Batch Inputs**

```bash
dx generate_batch_inputs \
	-itarget='target-(.*).bcf.gz' \
	--path='/adriano/target'
```
This command generates batch inputs. The input files are specified with a regular expression `-itarget='target-(.*).bcf.gz'`, and the input files are located in the specified path `--path='/adriano/target'`.

| batch ID | target |	target ID |
|----------|--------|-----------|
|0         | target0.vcf.gz |	project-GFX0638JzzB55yvP45JzYf6k:file-GQxBQK0JzzB88qb5v0ZpyK20 |
|1         | target1.vcf.gz |	project-GFX0638JzzB55yvP45JzYf6k:file-GQxBQK0JzzB88qb5v0ZpyK21 |
|2         | target2.vcf.gz |	project-GFX0638JzzB55yvP45JzYf6k:file-GQxBQK0JzzB88qb5v0ZpyK22 |

**Prepare reference panel for Selphi imputation**
```bash
for CHR in {1..22}; do
    file_id=$(dx find data --name chr${CHR}_ukb_100k.xsi --brief)
    file_id=${file_id#*:}  # Extract the file ID by removing the project ID prefix
    dx run selphi-imputation \
        -iprepare_reference=True \
        -iref_source_xsi="$file_id" \
        -irefpanel="/adriano/reference/chr${CHR}_ukb_100k" \
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
   -irefpanel='/path/to/reference/prefix-name-no-extention' \
   -imap='file-id-of-map-file-to-use' \
   -icores=10 \
   -ioutvcf='/path/to/outuput/prefix-name-no-extention'
   --instance-type='mem2_ssd1_v2_x16' \
   --priority low \
   --name "selphi-imputation test"
```

**Important notes for COST**

Make sure to choose the correct dna nexus instance in dxapp.json and choose the correct number of cores and run selphi in spot mode for optimal cost efficiency. And also this will relate to the target file size. it should be smaller with smaller instances.

<!--
TODO: This app directory was automatically generated by dx-app-wizard;
please edit this Readme.md file to include essential documentation about
your app that would be helpful to users. (Also see the
Readme.developer.md.) Once you're done, you can remove these TODO
comments.

For more info, see https://documentation.dnanexus.com/developer.
-->