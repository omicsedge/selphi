# selphi-imputation_kuk-optimizing
Refactoring &amp; optimizing Abdallah's &amp; Adriano's `rd-imputation-selphi` code


<br>

## Source code and data from Adriano and Abdallah

Relevant | Date | Source | Comment
--- | --- | --- | ---
:heavy_check_mark: | July 19th 2022 | https://s3.console.aws.amazon.com/s3/buckets/imputation-project?region=us-east-1&prefix=shared/selphi_fdraft/&showversions=false | [Aby] _All data included, but you just need to change the path for the genetic map file._
:white_circle: | July 28th 2022 | https://s3.console.aws.amazon.com/s3/buckets/imputation-project?region=us-east-1&prefix=shared/imputation_prj_versions/1/ | [Aby] _Done re structuring. Also you will find chr1 and chr20 already prepared for selphi imputation. For results checking, you have to add beagle imputed 292 data for chr20 because we didn't do them yet. You will find a notebook that convert it to zip array that is used in check_results.ipynb_ 
