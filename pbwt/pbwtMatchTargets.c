#include "pbwt.h"
#include <sys/stat.h>
#include <zip.h>

typedef struct MatchArrayStruct {
  // store matches for one target at one site //
  int fl; /* maximum number of matches kept*/
  int N; /* number of matches */
  int * haplotypes; /* array of int haplotype indices */
  int * lengths; /* array of int match lengths */
}
MatchArray;

typedef struct TargetMatchArrayStruct {
  // store matches for one target at all sites //
  int N; /* number of sites */
  int * hap_totals; /* array of total matches for haplotype */
  Array matchArrays; /* array of match arrays */
}
TargetMatchArray;

static void clearMatchArray(MatchArray * ma) {
  if (ma -> haplotypes) free(ma -> haplotypes);
  if (ma -> lengths) free(ma -> lengths);
}

static void TargetMatchArrayInit(TargetMatchArray * tma, int N, int M, int fl) {
  // Create array to store all matches for haplotype
  int i;
  MatchArray * ma;
  tma -> N = N;
  tma -> hap_totals = myalloc(M, int);
  for (i = 0; i < M; i++) tma -> hap_totals[i] = 0;
  tma -> matchArrays = arrayCreate(N, MatchArray);
  for (i = 0; i < N; i++) {
    ma = arrayp(tma -> matchArrays, i, MatchArray);
    ma -> fl = fl;
    ma -> N = 0;
  }
}

static void TargetMatchArrayDestroy(TargetMatchArray * tma) {
  if (tma -> hap_totals) free(tma -> hap_totals);
  if (tma -> matchArrays) {
    for (int i = 0; i < tma -> N; i++)
      clearMatchArray(arrayp(tma -> matchArrays, i, MatchArray));
    arrayDestroy(tma -> matchArrays);
  }
}

typedef struct SparseCscStruct {
  // store data for csc matrix format //
  int M; /* number of rows*/
  int N; /* number of columns*/
  int nnz; /* number of values */
  int * data; /* array of int, length nnz */
  int * indices; /* array of int, length nnz */
  int * indptr; /* array of int, length N + 1 */
}
SparseCsc;

static SparseCsc * SparseCscCreate(int M, int N, int nnz) {
  SparseCsc * csc = myalloc(1, SparseCsc);
  csc -> M = M;
  csc -> N = N;
  csc -> nnz = nnz;
  if (nnz > 0) {
    csc -> data = myalloc(nnz, int);
    csc -> indices = myalloc(nnz, int);
  }
  return csc;
}

static void SparseCscDestroy(SparseCsc * csc) {
  if (csc -> indptr) free(csc -> indptr);
  if (csc -> data) free(csc -> data);
  if (csc -> indices) free(csc -> indices);
  free(csc);
}

static void addMatch(int haplotype, int length, int var, TargetMatchArray * tma) {
  // Add match to array, keeping longest matches
  // Use sort distance as tie breaker for equally long matches
  tma -> hap_totals[haplotype] += length;
  int total = tma -> hap_totals[haplotype];

  MatchArray * ma = arrayp(tma -> matchArrays, var, MatchArray);
  int j = ma -> N - 1;
  if (ma -> N < ma -> fl) ma -> N++;
  int * haplotypes = myalloc(ma -> N, int);
  int * lengths = myalloc(ma -> N, int);

  while (j >= 0 && ma -> lengths[j] < length) {
    if (j + 1 < ma -> N) {
      lengths[j + 1] = ma -> lengths[j];
      haplotypes[j + 1] = ma -> haplotypes[j];
    }
    j--;
  }
  while (j >= 0 && ma -> lengths[j] == length && tma -> hap_totals[ma -> haplotypes[j]] <= total) {
    if (j + 1 < ma -> N) {
      lengths[j + 1] = ma -> lengths[j];
      haplotypes[j + 1] = ma -> haplotypes[j];
    }
    j--;
  }
  if (j + 1 < ma -> N) {
    lengths[j + 1] = length;
    haplotypes[j + 1] = haplotype;
  }
  while (j >= 0) {
    lengths[j] = ma -> lengths[j];
    haplotypes[j] = ma -> haplotypes[j];
    j--;
  }
  clearMatchArray(ma);
  ma -> lengths = lengths;
  ma -> haplotypes = haplotypes;
}

static SparseCsc * targetMatchesToCsc(TargetMatchArray * tma, int N, int M) {
  int i, j, n, nnz = 0;
  MatchArray * ma;

  // Backwards pass filtering matches per variant
  TargetMatchArray * ftma = myalloc(1, TargetMatchArray);
  TargetMatchArrayInit(ftma, N, M, 60);
  for (i = N - 1; i >= 0; i--) {
    ma = arrayp(tma -> matchArrays, i, MatchArray);
    for (j = ma -> N - 1; j >= 0; j--) addMatch(ma -> haplotypes[j], ma -> lengths[j], i, ftma);
  }

  int * indptr;
  indptr = myalloc(N + 1, int);
  for (i = 0; i < N; i++) {
    indptr[i] = nnz;
    nnz += arrayp(ftma -> matchArrays, i, MatchArray) -> N;
  }
  indptr[N] = nnz;

  SparseCsc * csc = SparseCscCreate(M, N, nnz);
  csc -> indptr = indptr;
  for (i = 0, n = 0; i < N; i++) {
    ma = arrayp(ftma -> matchArrays, i, MatchArray);
    for (j = 0; j < ma -> N; j++) {
      csc -> indices[n] = ma -> haplotypes[j];
      csc -> data[n] = ma -> lengths[j];
      n++;
    }
  }
  TargetMatchArrayDestroy(ftma);
  free(ftma);
  return csc;
}

static char * get_prefix(PBWT * p, int index) {
  char * sample_id = sampleName(sample(p, index));
  int strL = strlen(sample_id) + 3;
  char * prefix;
  prefix = (char * ) malloc(strL);
  snprintf(prefix, strL, "%s_%d", sample_id, (index % 2));
  return (char * ) prefix;
}

static void writeSparseCsc(char * filename, SparseCsc * csc) {
  int errorp, i;
  struct stat st = {0};
  if (stat(filename, & st) == 0) remove(filename);

  zip_t * archive = zip_open(filename, ZIP_CREATE, & errorp);

  char format[132];
  snprintf(format, 132,
    "\x93NUMPY\x01%cv%c{'descr': '|S3', 'fortran_order': False, 'shape': (), }%*s\ncsc",
    0, 0, 62, "");
  zip_file_add(archive, "format.npy",
    zip_source_buffer(archive, format, 131, 0), ZIP_FL_OVERWRITE);

  int shape[] = {csc -> M, csc -> N};
  char shape_buf[136];
  snprintf(shape_buf, 129,
    "\x93NUMPY\x01%cv%c{'descr': '<i4', 'fortran_order': False, 'shape': (2,), }%*s\n",
    0, 0, 60, "");
  memcpy(shape_buf + 128, shape, 8);
  zip_file_add(archive, "shape.npy",
    zip_source_buffer(archive, shape_buf, 136, 0), ZIP_FL_OVERWRITE);

  int ptr_size = 128 + (sizeof(int) * (csc -> N + 1));
  int ptrL = 60 - (int) log10(csc -> N + 1);
  char * indptr = malloc(ptr_size);
  snprintf(indptr, 129,
    "\x93NUMPY\x01%cv%c{'descr': '<i4', 'fortran_order': False, 'shape': (%d,), }%*s\n",
    0, 0, csc -> N + 1, ptrL, "");
  memcpy(indptr + 128, csc -> indptr, sizeof(int) * (csc -> N + 1));
  zip_file_add(archive, "indptr.npy",
    zip_source_buffer(archive, indptr, ptr_size, 0), ZIP_FL_OVERWRITE);

  int strL = 60 - (int) log10(csc -> nnz);
  char header[129];
  snprintf(header, 129, 
    "\x93NUMPY\x01%cv%c{'descr': '<i4', 'fortran_order': False, 'shape': (%d,), }%*s\n",
    0, 0, csc -> nnz, strL, "");
  int arr_size = sizeof(int) * csc -> nnz;

  char * indices = malloc(128 + arr_size);
  memcpy(indices, header, 128);
  memcpy(indices + 128, csc -> indices, arr_size);
  zip_file_add(archive, "indices.npy", 
    zip_source_buffer(archive, indices, 128 + arr_size, 0), ZIP_FL_OVERWRITE);

  char * data = malloc(128 + arr_size);
  memcpy(data, header, 128);
  memcpy(data + 128, csc -> data, arr_size);
  zip_file_add(archive, "data.npy", 
    zip_source_buffer(archive, data, 128 + arr_size, 0), ZIP_FL_OVERWRITE);

  zip_close(archive);
  free(indptr);
  free(indices);
  free(data);
}

static void write_npz(char * prefix, TargetMatchArray * tma, int N, int M) {
  // Expand matches
  SparseCsc * csc = targetMatchesToCsc(tma, N, M);
  // Create npz file
  char tmp_name[84];
  snprintf(tmp_name, 84, "parallel_haploid_mat_%s.npz", prefix);
  writeSparseCsc(tmp_name, csc);
  SparseCscDestroy(csc);
}

static void pbwtMatchTargets(PBWT * p, int minL, int nRefHaps) {
  int var, hap, rHap, tHap, ia, ib, ir, id, i0, dmin, dmin_, tgt_idx;
  PbwtCursor * u = pbwtCursorCreate(p, TRUE, TRUE);
  int nVar = p -> N;
  int nAllHaps = u -> M;
  int nTargetHaps = nAllHaps - nRefHaps;
  int * refHaps;
  int * targetHaps;
  printf(" [pbwt]: %d reference haplotypes, %d target haplotypes\n", nRefHaps, nTargetHaps);

  Array matchArrays = arrayCreate(nTargetHaps, TargetMatchArray);
  for (tHap = 0; tHap < nTargetHaps; tHap++)
    TargetMatchArrayInit(arrayp(matchArrays, tHap, TargetMatchArray), nVar, nRefHaps, 450);

  printf(" [pbwt]: Matching target haplotypes to reference panel\n");
  for (var = 0; var < nVar; ++var) {
    if (var < minL) {
      pbwtCursorForwardsReadAD(u, var);
      continue;
    }
    refHaps = myalloc(nRefHaps, int);
    targetHaps = myalloc(nTargetHaps, int);
    for (hap = 0, rHap = 0, tHap = 0; hap < nAllHaps; hap++) {
      if (u -> a[hap] >= nRefHaps) {
        targetHaps[tHap] = hap;
        tHap++;
      } else {
        refHaps[rHap] = hap;
        rHap++;
      }
    }
    for (tHap = 0; tHap < nTargetHaps; tHap++) {
      rHap = 0;
      tgt_idx = u -> a[targetHaps[tHap]] - nRefHaps;
      ib = targetHaps[tHap];
      while (ib > refHaps[rHap] && rHap < nRefHaps) rHap++;
      for (ir = rHap - 1, i0 = ib, dmin_ = 0; ir >= 0; ir--) {
        ia = refHaps[ir];
        for (id = i0; id > ia; id--)
          if (u -> d[id] > dmin_) dmin_ = u -> d[id];
        if (dmin_ <= var - minL)
          if ((u -> y[ib] != u -> y[ia]) || (var >= nVar - 1))
            addMatch(u -> a[ia], var - dmin_, dmin_,
              arrayp(matchArrays, tgt_idx, TargetMatchArray));
        i0 = ia;
      }
      ia = targetHaps[tHap];
      for (ib = ia + 1, dmin = 0; ib <= refHaps[(nRefHaps - 1)]; ib++) {
        if (u -> d[ib] > dmin) dmin = u -> d[ib];
        if (dmin <= var - minL)
          if ((u -> y[ib] != u -> y[ia]) || (var >= nVar - 1))
            if (u -> a[ib] < nRefHaps)
              addMatch(u -> a[ib], var - dmin, dmin,
                arrayp(matchArrays, tgt_idx, TargetMatchArray));
      }
    }
    free(refHaps);
    free(targetHaps);
    pbwtCursorForwardsReadAD(u, var);
    if (var % 100 == 0) printProgress((double) var / nVar);
  }
  printProgress(1.0);
  printf("\n [pbwt]: Writing output\n");
  TargetMatchArray * tma;
  for (tHap = nRefHaps; tHap < nAllHaps; tHap++) {
    tgt_idx = tHap - nRefHaps;
    tma = arrayp(matchArrays, tgt_idx, TargetMatchArray);
    write_npz(get_prefix(p, tHap), tma, nVar, nRefHaps);
    TargetMatchArrayDestroy(tma);
    printProgress((double)(tgt_idx + 1) / nTargetHaps);
  }
  printf("\n");
  pbwtCursorDestroy(u);
  arrayDestroy(matchArrays);
}

void referenceMatch(PBWT * pTargets, char * fileNameRoot, int minL) {
  if (pTargets -> M % 2) die("requires that M = %d is even", pTargets -> M);
  if (!pTargets || !pTargets -> yz || !pTargets -> sites)
    die(" [pbwt]: referenceMatch called without targets pbwt with sites");

  PBWT * pRef = pbwtReadAll(fileNameRoot);
  int nRefHaps = pRef -> M;
  if (!pRef -> sites) die("%s reference panel has no sites", fileNameRoot);
  if (strcmp(pTargets -> chrom, pRef -> chrom))
    die(" [pbwt]: mismatching chrom in reference panel: old %s, ref %s",
      pTargets -> chrom, pRef -> chrom);

  PBWT * pAll = pbwtMerge2(pRef, pTargets);
  pbwtDestroy(pRef);
  printf(" [pbwt]: Merged PBWT has %d haplotypes and %d sites\n", pAll -> M, pAll -> N);

  pbwtMatchTargets(pAll, minL, nRefHaps);
  pbwtDestroy(pAll);
}
