/*  File: pbwt.h
 *  Original code authored by: Richard Durbin (rd@sanger.ac.uk)
 *  Copyright (C) Genome Research Limited, 2013-
 *-------------------------------------------------------------------
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------
 * Description: header file for pbwt package
 *
 * This is a slimmed down version of the original pbwt repository 
 * with added functionality for getting all matches of a certain length
 * between target haplotypes and all haplotypes in a reference panel.
 *-------------------------------------------------------------------
 */

#include "utils.h"

const static int pbwtMajorVersion = 3,
  pbwtMinorVersion = 0;

const char * pbwtCommitHash(void);
const char * pbwtHtslibVersionString(void);

/* data types */

typedef unsigned char uchar;

typedef struct PBWTstruct {
  int N; /* number of sites */
  int M; /* number of samples */
  char * chrom; /* chromosome name */
  Array sites; /* array of Site */
  Array samples; /* array of int index into global samples */
  Array yz; /* compressed PBWT array of uchar */
  int * aFstart, * aFend; /* start and end a[] index arrays for forwards cursor */
  Array zz; /* compressed reverse PBWT array of uchar */
  int * aRstart, * aRend; /* start and end a[] index arrays for reverse cursor */
  /* NB aRend is the lexicographic sort order for the data, and aFend the reverse lex order */
  /* probably it is optimal to have aFstart == aRend and vice versa: to be done */
  Array zMissing; /* compressed array of uchar - natural not sort order */
  Array missingOffset; /* of long, site index into zMissing, 0 if no missing data at site */
  BOOL isRefFreq; /* some flags for the whole VCF */
  BOOL isUnphased;
}
PBWT;

/* philosophy is to be lazy about PBWT - only fill items for which we have info */

typedef struct SiteStruct {
  int x; /* position on chromosome */
  int varD; /* index in variationDict */
  double freq; /* frequency */
  double refFreq; /* frequency from reference used for last phasing or imputation */
  double imputeInfo; /* estimated r^2 from imputation */
}
Site;

typedef struct SampleStruct {
  int nameD; /* index in sampleDict */
  int father; /* index into samples */
  int mother; /* index into samples */
  int popD; /* index in populationDict */
  BOOL isMale; /* treat X chromosome as haploid */
  BOOL isFemale; /* treat X chromosome as diploid and ignore Y */
}
Sample;

typedef struct {
  /* data structure for moving forwards - doesn't know PBWT */
  int M;
  Array z; /* packed byte array; if zero y needs loading from elsewhere */
  long n; /* position in packed byte array */
  BOOL isBlockEnd; /* TRUE if n is at end of next block, FALSE if at start */
  uchar * y; /* current value in sort order */
  int c; /* number of 0s in y */
  int * a; /* index back to original order */
  int * d; /* location of last match */
  int * u; /* number of 0s up to and including this position */
  int * b; /* for local operations - no long term meaning */
  int * e; /* for local operations - no long term meaning */
  long nBlockStart; /* u->n at start of block encoding current u->y */
}
PbwtCursor;

/* pbwtMain.c */
extern FILE * logFile; /* log file pointer */

/* pbwtCore.c */
extern BOOL isCheck; /* when TRUE carry out various checks */
extern BOOL isStats; /* when TRUE report stats in various places */
extern DICT * variationDict; 
void pbwtInit(void);
PBWT * pbwtCreate(int M, int N); /* OK to have N == 0 and set p->N later if not known now */
void pbwtDestroy(PBWT * p);
uchar ** pbwtHaplotypes(PBWT * p);
/* low level operations on packed PBWT, argument yzp in these calls */
#define Y_SENTINEL 2 /* needed to pack efficiently */
int pack3(uchar * yp, int M, uchar * yzp); /* pack M values from yp into yzp */
int pack3arrayAdd(uchar * yp, int M, Array ayz); /* normally use this one */
int unpack3(uchar * yzp, int M, uchar * yp, int * n0); /* unpack M values from yzp into yp, return number of bytes used from yzp, if (n0) write number of 0s into *n0 */
int packCountReverse(uchar * yzp, int M); /* return number of bytes to reverse one position */
int extendMatchForwards(uchar * yzp, int M, uchar x, int * f, int * g); /* move hit interval f,g) forwards one position, matching x */
int extendPackedForwards(uchar * yzp, int M, int * f, uchar * zp); /* move f forwards one position */
int extendPackedBackwards(uchar * yzp, int M, int * f, int c, uchar * zp); /* move f backwards one position - write value into *zp if zp non-zero */
/* cursors */
PbwtCursor * pbwtNakedCursorCreate(int M, int * aInit);
PbwtCursor * pbwtCursorCreate(PBWT * p, BOOL isForwards, BOOL isStart);
void pbwtCursorDestroy(PbwtCursor * u);
void pbwtCursorForwardsA(PbwtCursor * u); /* algorithm 1 in the manuscript */
void pbwtCursorBackwardsA(PbwtCursor * u); /* undo algorithm 1 */
void pbwtCursorForwardsAD(PbwtCursor * u, int k); /* algorithm 2 in the manuscript */
void pbwtCursorCalculateU(PbwtCursor * x); /* calculate u required for CursorMap */
void pbwtCursorForwardsRead(PbwtCursor * u); /* move forwards and read (unless at end) */
void pbwtCursorForwardsReadAD(PbwtCursor * u, int k);
void pbwtCursorReadBackwards(PbwtCursor * u); /* read and move backwards (unless at start) */
void pbwtCursorWriteForwards(PbwtCursor * u); /* write then move forwards */
void pbwtCursorWriteForwardsAD(PbwtCursor * u, int k);
void pbwtCursorToAFend(PbwtCursor * u, PBWT * p); /* utility to copy final u->a to p->aFend */
void pbwtCursorForwardsAPacked(PbwtCursor * u); /* faster version, when have read y and set u->nBlockStart */
static inline int pbwtCursorMap(PbwtCursor * u, int x, int i) {
  return x ? u -> c + i - u -> u[i] : u -> u[i];
}
static inline int pbwtCursorMapDplus(PbwtCursor * u, int x, int i, int dplus) {
  for (; i < u -> M && u -> y[i] != x; ++i)
    if (u -> d[i] > dplus) dplus = u -> d[i];
  return dplus;
}
static inline int pbwtCursorMapDminus(PbwtCursor * u, int x, int i, int dminus) {
  for (--i; i >= 0 && u -> y[i] != x; --i)
    if (u -> d[i] > dminus) dminus = u -> d[i];
  return dminus;
}
PBWT * pbwtSelectSites(PBWT * pOld, Array sites, BOOL isKeepOld);

/* pbwtSample.c */
void sampleInit(void);
void sampleDestroy(void);
int sampleAdd(char * name, char * father, char * mother, char * pop);
Sample * sample(PBWT * p, int i); /* give back Sample structure for sample i from p */
char * sampleName(Sample * s);
char * popName(Sample * s); /* give back population name for sample i */
PBWT * pbwtSelectSamples(PBWT * pOld, FILE * fp);

/* pbwtIO.c */
extern int nCheckPoint; /* if set non-zero write pbwt and sites files every n sites when parsing external files */
void pbwtCheckPoint(PbwtCursor * u, PBWT * p);
void pbwtWriteSites(PBWT * p, FILE * fp);
void pbwtWriteAll(PBWT * p, char * fileNameRoot);
Array pbwtReadSitesFile(FILE * fp, char ** chrom);
Array pbwtReadSamplesFile(FILE *fp);
PBWT * pbwtReadAll(char * fileNameRoot); /* reads .pbwt, .sites, .samples, .missing  */

/* pbwtHtslib.c */
PBWT * pbwtReadVcfGT(char * filename); /* read GTs from vcf/bcf using htslib */

/* pbwtMatchTargets.c - functions for SelfDecode */
void referenceMatch(PBWT * p, char * fileNameRoot, int minL);

/* pbwtMerge.c - modified for SelfDecode */
PBWT * pbwtMerge2(PBWT * p, PBWT * q);

/******************* end of file *******************/
