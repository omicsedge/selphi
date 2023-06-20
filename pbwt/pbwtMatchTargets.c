#include "pbwt.h"
#include <sys/stat.h>
#include <zip.h>

static char *get_prefix(PBWT *p, int index) {
  // Assume sample id is not more than 50 bytes
  char *sample_id = sampleName(sample(p, index));
  int strL = 54;
  char *prefix;
  prefix = (char *)malloc(strL);
  snprintf(prefix, strL, "%s_%d", sample_id, (index % 2));
  return (char *)prefix;
}

static char *get_path(char *prefix, char *basename) {
  // Longest basename is 11 bytes
  // Assume sample id is not more than 50 bytes
  int strL = 68;
  char *filename;
  filename = (char *)malloc(strL);
  snprintf(filename, strL, "%s/%s", prefix, basename);
  return (char *)filename;
}

static void write_npy_header(FILE *f, char *dtype, char *shape) {
  fwrite("\x93NUMPY\x01\x00v", 1, 10, f);
  fprintf(f, "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }", dtype, shape);
  fprintf(f, "%*s\n", 127 - (int)ftell(f), "");
}

static void init_array_file(char *filename) {
  FILE *f = fopen(filename, "wb");
  if (f == NULL) die(" [pbwt]: Could not create %s\n", filename);
  write_npy_header(f, "<i4", "");
  fclose(f);
}

static void create_npz_files(char *prefix, int nVar, int nRefHaps) {
  struct stat st = {0};
  if (stat(prefix, &st) == -1) mkdir(prefix, 0700);
  // Create sparse matrix files and write headers
  FILE *f = fopen(get_path(prefix, "format.npy"), "wb");
  if (f == NULL) die(" [pbwt]: Could not create %s\n", get_path(prefix, "format.npy"));
  write_npy_header(f, "|S3", "");
  fputs("coo", f);
  fclose(f);

  f = fopen(get_path(prefix, "shape.npy"), "wb");
  if (f == NULL) die(" [pbwt]: Could not create %s\n", get_path(prefix, "shape.npy"));
  write_npy_header(f, "<i4", "2,");
  int shape[] = {nRefHaps, nVar};
  fwrite(shape, sizeof(int), 2, f);
  fclose(f);

  init_array_file(get_path(prefix, "col.npy"));
  init_array_file(get_path(prefix, "row.npy"));
  init_array_file(get_path(prefix, "data.npy"));
}

static void write_array(char *filename, int *data, int length) {
  int data_[length];
  int i;
  for (i = 0; i < length; i++)
    data_[i] = data[i];
  FILE *f = fopen(filename, "ab");
  if (f == NULL) die(" [pbwt]: Error opening %s\n", filename);
  fwrite(data_, sizeof(int), length, f);
  fclose(f);
}

static void write_data(char *prefix, int *cols,  int *rows,  int *data, int length) {
  write_array(get_path(prefix, "col.npy"), cols, length);
  write_array(get_path(prefix, "row.npy"), rows, length);
  write_array(get_path(prefix, "data.npy"), data, length);
}

static void update_npy_header(char *filename, int length) {
  FILE *f = fopen(filename, "r+b");
  if (f == NULL) die(" [pbwt]: Error opening %s\n", filename);
  fseek(f, 61, SEEK_SET);
  fprintf(f, "%d,), }", length);
  fclose(f);
}

static void close_npz(char *prefix, int length) {
  int errorp, i;
  char *files[] = {"format.npy", "shape.npy", "col.npy", "row.npy", "data.npy"};
  // Finish array headers
  for (i = 2; i < 5; i++) update_npy_header(get_path(prefix, files[i]), length);
  // Create npz file
  char tmp_name[84];
  snprintf(tmp_name, 84, "parallel_haploid_mat_%s.npz", prefix);
  struct stat st = {0};
  if (stat(tmp_name, &st) == 0) remove(tmp_name);
  zip_t *archive = zip_open(tmp_name, ZIP_CREATE, &errorp);
  zip_source_t * source;
  for (i = 0; i < 5; i++) {
    source = zip_source_file(archive, get_path(prefix, files[i]), 0, 0);
    zip_file_add(archive, files[i], source, ZIP_FL_OVERWRITE);
    zip_set_file_compression(archive, i, ZIP_CM_DEFLATE, 1);
  }
  zip_close(archive);
  // Clean up npy files
  for (i = 0; i < 5; i++) remove(get_path(prefix, files[i]));
  remove(prefix);
}

static void pbwtMatchTargets(PBWT * p, int minL, int nRefHaps) {
  int var, hap, rHap, tHap, ia, ib, ir, id, i0, dmin, dmin_, im;
  PbwtCursor * u = pbwtCursorCreate(p, TRUE, TRUE);
  int nVar = p -> N;
  int nAllHaps = u -> M;
  int nTargetHaps = nAllHaps - nRefHaps;
  int rArrSize = (int) (nRefHaps * 1.1 + 1);
  int tArrSize = (int) (nTargetHaps * 1.1 + 1);
  int *refHaps;
  int *targetHaps;
  int *matchHaps;
  int *matchVars;
  int *matchLens;
  int *matchCounts;
  matchCounts = myalloc(tArrSize, int);
  for (tHap = 0; tHap < nTargetHaps; tHap++) matchCounts[tHap] = 0;
  printf(" [pbwt]: %d reference haplotypes, %d target haplotypes\n", nRefHaps, nTargetHaps);

  for (tHap = nRefHaps; tHap < nAllHaps; tHap++)
    create_npz_files(get_prefix(p, tHap), nVar, nRefHaps);

  printf(" [pbwt]: Matching target haplotypes to reference panel\n");
  for (var = 0; var <= nVar; ++var) {
    if (var < minL) {
      pbwtCursorForwardsReadAD(u, var);
      continue;
    }
    refHaps = myalloc(rArrSize, int);
    targetHaps = myalloc(tArrSize, int);
    for (hap = 0, rHap = 0, tHap = 0; hap < nAllHaps; hap++) {
      if (u -> a[hap] >= nRefHaps) {
        targetHaps[tHap] = hap;
        tHap++;
      } else {
        refHaps[rHap] = hap;
        rHap++;
      }
    }
    for (tHap = 0; tHap < nTargetHaps; ++tHap) {
      matchHaps = myalloc(rArrSize, int);
      matchVars = myalloc(rArrSize, int);
      matchLens = myalloc(rArrSize, int);
      im = 0;
      rHap = 0;
      ib = targetHaps[tHap];
      while (ib > refHaps[rHap]) rHap++;
      for (ir = rHap - 1, i0 = ib, dmin_ = 0; ir >= 0; --ir) {
        ia = refHaps[ir];
        for (id = i0; id > ia; --id)
          if (u -> d[id] > dmin_) dmin_ = u -> d[id];
        if (dmin_ <= var - minL)
          if ((u -> y[ib] != u -> y[ia]) || (var == nVar)) {
            matchHaps[im] = u -> a[ia];
            matchVars[im] = dmin_;
            matchLens[im] = var - dmin_;
            im++;
          }
        i0 = ia;
      }
      ia = targetHaps[tHap];
      for (ib = ia + 1, dmin = 0; ib <= refHaps[(nRefHaps - 1)]; ++ib) {
        if (u -> d[ib] > dmin) dmin = u -> d[ib];
        if (dmin <= var - minL)
          if ((u -> y[ib] != u -> y[ia]) || (var == nVar))
            if (u -> a[ib] < nRefHaps) {
              matchHaps[im] = u -> a[ib];
              matchVars[im] = dmin;
              matchLens[im] = var - dmin;
              im++;
            }
      }
      if (im > 0) {
        write_data(get_prefix(p, u -> a[ia]), matchVars, matchHaps, matchLens, im);
        matchCounts[(u -> a[ia] - nRefHaps)] += im;
      }
      free(matchHaps);
      free(matchVars);
      free(matchLens);
    }
    free(refHaps);
    free(targetHaps);
    pbwtCursorForwardsReadAD(u, var);
  }
  printf(" [pbwt]: Compressing output\n");
  for (tHap = nRefHaps; tHap < nAllHaps; tHap++)
    close_npz(get_prefix(p, tHap), matchCounts[(tHap - nRefHaps)]);
  pbwtCursorDestroy(u);
  free(matchCounts);
}

void referenceMatch(PBWT * pTargets, char * fileNameRoot, int minL) {
  if (pTargets->M % 2) die("requires that M = %d is even", pTargets->M);
  if (!pTargets || !pTargets->yz || !pTargets->sites) 
    die(" [pbwt]: referenceMatch called without targets pbwt with sites");

  PBWT * pRef = pbwtReadAll(fileNameRoot);
  int nRefHaps = pRef->M;
  if (!pRef->sites) die("%s reference panel has no sites", fileNameRoot);
  if (strcmp(pTargets->chrom, pRef->chrom))
    die(" [pbwt]: mismatching chrom in reference panel: old %s, ref %s", pTargets->chrom, pRef->chrom);

  // reduce both down to the intersecting sites
  pRef = pbwtSelectSites(pRef, pTargets->sites, FALSE);
  pTargets = pbwtSelectSites(pTargets, pRef->sites, FALSE);
  if (!pTargets->N) die("no overlapping sites in reference panel");

  PBWT * pAll = pbwtMerge2(pRef, pTargets);
  pbwtDestroy(pRef);
  FILE *f = fopen("variants.txt", "wb");
  pbwtWriteSites(pAll, f);
  fclose(f);
  printf(" [pbwt]: Merged PBWT has %d haplotypes and %d sites\n", pAll->M, pAll->N);

  pbwtMatchTargets(pAll, minL, nRefHaps);
  pbwtDestroy(pAll);
}
