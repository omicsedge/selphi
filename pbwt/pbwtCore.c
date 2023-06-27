/*  File: pbwtCore.c
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
 * Description: core functions for pbwt package
 *
 * This is a slimmed down version of the original pbwt repository 
 * with added functionality for getting all matches of a certain length
 * between target haplotypes and all haplotypes in a reference panel.
 *-------------------------------------------------------------------
 */

#include "pbwt.h"

BOOL isCheck = FALSE;
BOOL isStats = FALSE;
DICT * variationDict; /* "xxx|yyy" where variation is from xxx to yyy in VCF */

static void pack3init(void); /* forward declaration */

/************** core pbwt options *********************/

void pbwtInit(void) {
  variationDict = dictCreate(32);
  pack3init();
  sampleInit();
}

PBWT * pbwtCreate(int M, int N) {
  PBWT * p = mycalloc(1, PBWT); /* cleared so elements default to 0 */

  p -> M = M;
  p -> N = N;
  p -> aFstart = myalloc(M, int);
  int i;
  for (i = 0; i < M; ++i) p -> aFstart[i] = i;

  return p;
}

void pbwtDestroy(PBWT * p) {
  if (p -> chrom) free(p -> chrom);
  if (p -> sites) arrayDestroy(p -> sites);
  if (p -> samples) arrayDestroy(p -> samples);
  if (p -> yz) arrayDestroy(p -> yz);
  if (p -> zz) arrayDestroy(p -> zz);
  if (p -> aFstart) free(p -> aFstart);
  if (p -> aFend) free(p -> aFend);
  if (p -> aRstart) free(p -> aRstart);
  if (p -> aRend) free(p -> aRend);
  if (p -> missingOffset) arrayDestroy(p -> missingOffset);
  if (p -> zMissing) arrayDestroy(p -> zMissing);
  free(p);
}

/*************** make haplotypes ******************/

uchar ** pbwtHaplotypes(PBWT * p) {
  int M = p -> M;
  int i, j, n = 0;
  int * a;
  PbwtCursor * u = pbwtCursorCreate(p, TRUE, TRUE);
  uchar ** hap = myalloc(M, uchar * );

  for (i = 0; i < M; ++i) hap[i] = myalloc(p -> N, uchar);

  for (i = 0; i < p -> N; ++i) {
    for (j = 0; j < M; ++j) hap[u -> a[j]][i] = u -> y[j];
    pbwtCursorForwardsRead(u);
  }
  pbwtCursorDestroy(u);
  return hap;
}

/****************** Y compression and decompression *************************/
/****************** low level - does not know about PBWT structures *********/

/* pack3 is a three level run length encoding: n times value
   yp & 0x80 = value
   yp & 0x40 == 0 implies n = yp & 0x3f
   yp & 0x40 == 1 implies
     yp & 0x20 == 0 implies n = (yp & 0x1f) << 6
     yp & 0x20 == 1 implies n = (yp & 0x1f) << 11
   This allows coding runs of length up to 64 * 32 * 32 = 64k in 3 bytes.
   Potential factor of ~1000 over a bit array.
   Build a lookup to avoid conditional operation in uncompression.
*/

static int p3decode[128];
#define ENCODE_MAX1 64 /* ~64 */
#define ENCODE_MAX2 ((95 - 63) << 6) /* ~1k - is this 32 or 31?*/
#define ENCODE_MAX3 ((127 - 96) << 11) /* ~64k - ditto */

static void pack3init(void) {
  int n;
  for (n = 0; n < 64; ++n) p3decode[n] = n;
  for (n = 64; n < 96; ++n) p3decode[n] = (n - 64) << 6;
  for (n = 96; n < 128; ++n) p3decode[n] = (n - 96) << 11;
}

static inline int pack3Add(uchar yy, uchar * yzp, int n) {
  uchar * yzp0 = yzp;

  yy <<= 7; /* first move the actual symbol to the top bit */

  while (n >= ENCODE_MAX3) {
    * yzp++ = yy | 0x7f;
    n -= ENCODE_MAX3;
  }
  if (n >= ENCODE_MAX2) {
    * yzp++ = yy | 0x60 | (n >> 11);
    n &= 0x7ff;
  }
  if (n >= ENCODE_MAX1) {
    * yzp++ = yy | 0x40 | (n >> 6);
    n &= 0x3f;
  }
  if (n) {
    * yzp++ = yy | n;
  }
  return yzp - yzp0;
}

static int pack3(uchar * yp, int M, uchar * yzp) {
  int m = 0, m0;
  uchar * yzp0 = yzp;
  uchar ym;

  while (m < M) /* NB this relies on the M'th character not being 0 or 1 */ {
    ym = * yp++;
    m0 = m++;
    while ( * yp == ym) {
      ++m;
      ++yp;
    }
    yzp += pack3Add(ym, yzp, m - m0);
  }
  return yzp - yzp0;
}

int pack3arrayAdd(uchar * yp, int M, Array ayz) {
  long max = arrayMax(ayz);
  arrayExtend(ayz, max + M); /* ensure enough space to copy into */
  int n = pack3(yp, M, arrp(ayz, max, uchar));
  arrayMax(ayz) = max + n;
  return n;
}

int unpack3(uchar * yzp, int M, uchar * yp, int * n0) {
  int m = 0;
  uchar * yzp0 = yzp;
  uchar yz;
  int n;

  if (n0) * n0 = 0;
  while (m < M) {
    yz = * yzp++;
    n = p3decode[yz & 0x7f];
    m += n;
    yz = yz >> 7;
    if (n0 && !yz) * n0 += n;
    if (n > 63) /* only call memset if big enough */ {
      memset(yp, yz, n);
      yp += n;
    } else
      while (n--) * yp++ = yz;
  }
  if (isCheck && m != M)
    die("mismatch m %d != M %d in unpack3 after unpacking %d\n", m, M, yzp - yzp0);

  return yzp - yzp0;
}

/************ block extension algorithms, updating whole arrays ************/
/* we could do these also on the packed array with memcpy */

PbwtCursor * pbwtNakedCursorCreate(int M, int * aInit) {
  PbwtCursor * u = mycalloc(1, PbwtCursor);
  int i;

  u -> M = M;
  u -> y = myalloc(M + 1, uchar);
  u -> y[M] = Y_SENTINEL;
  u -> a = myalloc(M, int);
  if (aInit) memcpy(u -> a, aInit, M * sizeof(int));
  else
    for (i = 0; i < M; ++i) u -> a[i] = i;
  u -> b = myalloc(M, int);
  u -> d = mycalloc(M + 1, int);
  u -> d[0] = 1;
  u -> d[u -> M] = 1; /* sentinel values */
  u -> e = myalloc(M + 1, int);
  u -> u = myalloc(M + 1, int);
  return u;
}

PbwtCursor * pbwtCursorCreate(PBWT * p, BOOL isForwards, BOOL isStart) {
  if (isForwards && !p -> yz) p -> yz = arrayCreate(1 << 20, uchar);
  if (!isForwards && !p -> zz) p -> zz = arrayCreate(1 << 20, uchar);
  PbwtCursor * u;
  if (isForwards && isStart) u = pbwtNakedCursorCreate(p -> M, p -> aFstart);
  else if (isForwards && !isStart) u = pbwtNakedCursorCreate(p -> M, p -> aFend);
  else if (!isForwards && isStart) u = pbwtNakedCursorCreate(p -> M, p -> aRstart);
  else if (!isForwards && !isStart) u = pbwtNakedCursorCreate(p -> M, p -> aRend);
  if (isForwards) u -> z = p -> yz;
  else u -> z = p -> zz;
  if (isStart)
    if (arrayMax(u -> z)) {
      u -> nBlockStart = 0;
      u -> n = unpack3(arrp(u -> z, 0, uchar), p -> M, u -> y, & u -> c);
      u -> isBlockEnd = TRUE;
    }
  else {
    u -> n = 0;
    u -> isBlockEnd = FALSE;
  } else /* isEnd */ {
    u -> n = arrayMax(u -> z);
    u -> isBlockEnd = FALSE;
  }
  return u;
}

void pbwtCursorDestroy(PbwtCursor * u) {
  free(u -> y);
  free(u -> a);
  free(u -> b);
  free(u -> d);
  free(u -> e);
  free(u -> u);
  free(u);
}

void pbwtCursorForwardsA(PbwtCursor * x) {
  int u = 0, v = 0;
  int i;

  for (i = 0; i < x -> M; ++i)
    if (x -> y[i] == 0)
      x -> a[u++] = x -> a[i];
    else /* y[i] == 1, since bi-allelic */
      x -> b[v++] = x -> a[i];

  memcpy(x -> a + u, x -> b, v * sizeof(int));
}

void pbwtCursorForwardsAD(PbwtCursor * x, int k) {
  int u = 0, v = 0;
  int i;
  int p = k + 1;
  int q = k + 1;

  for (i = 0; i < x -> M; ++i) {
    if (x -> d[i] > p) p = x -> d[i];
    if (x -> d[i] > q) q = x -> d[i];
    if (x -> y[i] == 0) /* NB x[a[i]] = y[i] in manuscript */ {
      x -> a[u] = x -> a[i];
      x -> d[u] = p;
      ++u;
      p = 0;
    } else /* y[i] == 1, since bi-allelic */ {
      x -> b[v] = x -> a[i];
      x -> e[v] = q;
      ++v;
      q = 0;
    }
  }

  memcpy(x -> a + u, x -> b, v * sizeof(int));
  memcpy(x -> d + u, x -> e, v * sizeof(int));
  x -> d[0] = k + 2;
  x -> d[x -> M] = k + 2; /* sentinels */
}

/* We need to be careful about isBlockEnd in the routines below because when reading
   forwards we will have ->n naturally set after the end of the last block read,
   whereas when reading backwards it will naturally be at the start of the last block read.
   Also when writing it is at the start of the next block, because we don't yet have the contents.
*/

/***************************************************/

static void pbwtCursorForwardsAPacked(PbwtCursor * u) {
  int c = 0, m = 0, n;
  uchar * zp = arrp(u -> z, u -> nBlockStart, uchar), * zp0 = zp, z;
  while (m < u -> M) {
    z = * zp++;
    n = p3decode[z & 0x7f];
    z >>= 7;
    if (z)
      memcpy(u -> b + (m - c), u -> a + m, n * sizeof(int));
    else {
      memcpy(u -> e + c, u -> a + m, n * sizeof(int));
      c += n;
    }
    m += n;
  }
  if (m != u -> M) die("error in forwardsAPacked()");

  memcpy(u -> a, u -> e, u -> c * sizeof(int));
  memcpy(u -> a + u -> c, u -> b, (u -> M - u -> c) * sizeof(int));
}

void pbwtCursorForwardsRead(PbwtCursor * u) /* move forwards and read (unless at end) */ {
  pbwtCursorForwardsAPacked(u);
  if (!u -> isBlockEnd && u -> n < arrayMax(u -> z)) /* move to end of previous block */ {
    u -> nBlockStart = u -> n;
    u -> n += unpack3(arrp(u -> z, u -> n, uchar), u -> M, u -> y, 0);
  }
  if (u -> n < arrayMax(u -> z)) {
    u -> nBlockStart = u -> n;
    u -> n += unpack3(arrp(u -> z, u -> n, uchar), u -> M, u -> y, & u -> c); /* read this block */
    u -> isBlockEnd = TRUE;
  } else
    u -> isBlockEnd = FALSE; /* couldn't read in block and go to end */
}

void pbwtCursorForwardsReadAD(PbwtCursor * u, int k) /* AD version of the above */ {
  pbwtCursorForwardsAD(u, k);
  if (!u -> isBlockEnd && u -> n < arrayMax(u -> z)) /* move to end of previous block */ {
    u -> nBlockStart = u -> n;
    u -> n += unpack3(arrp(u -> z, u -> n, uchar), u -> M, u -> y, 0);
  }
  if (u -> n < arrayMax(u -> z)) {
    u -> nBlockStart = u -> n;
    u -> n += unpack3(arrp(u -> z, u -> n, uchar), u -> M, u -> y, & u -> c); /* read this block */
    u -> isBlockEnd = TRUE;
  } else
    u -> isBlockEnd = FALSE; /* couldn't read in block and go to end */
}

void pbwtCursorWriteForwards(PbwtCursor * u) {
  u -> n += pack3arrayAdd(u -> y, u -> M, u -> z);
  u -> isBlockEnd = FALSE;
  pbwtCursorForwardsA(u);
}

void pbwtCursorToAFend(PbwtCursor * u, PBWT * p) {
  if (!p -> aFend) p -> aFend = myalloc(p -> M, int);
  memcpy(p -> aFend, u -> a, p -> M * sizeof(int));
}

/***************************************************/

PBWT * pbwtFilterSites(PBWT * pOld, Array filter) {
  /* Provide array of 0s and 1s to filter sites from pbwt*/
  if (arrayMax(filter) != pOld->N) die("Filter is not the same size as pbwt");
  int i, j, newN;
  for (j = 0; j < pOld->N; ++j) newN += * arrp(filter, j, int);
  if (newN == pOld->N) return pOld;

  PBWT * pNew = pbwtCreate(pOld->M, 0);
  PbwtCursor * uOld = pbwtCursorCreate(pOld, TRUE, TRUE);
  PbwtCursor * uNew = pbwtCursorCreate(pNew, TRUE, TRUE);
  uchar * yseq = myalloc(pNew->M, uchar);

  pNew->sites = arrayCreate(newN, Site);

  for (j = 0; j < pOld->N; ++j) {
    if (* arrp(filter, j, int) == 1) {
      array(pNew->sites, pNew->N++, Site) = * arrp(pOld->sites, j, Site);
      for (i = 0; i < pOld->M; ++i) yseq[uOld->a[i]] = uOld->y[i];
      for (i = 0; i < pNew->M; ++i) uNew->y[i] = yseq[uNew->a[i]];
      pbwtCursorWriteForwards(uNew);
    }
    pbwtCursorForwardsRead(uOld);
    if (j % 1000 == 0) printProgress((double) j / pOld->N);
  }
  printProgress(1.0);
  printf("\n");

  pbwtCursorToAFend(uNew, pNew);
  pNew->chrom = pOld->chrom;
  pOld->chrom = 0;
  pNew->samples = pOld->samples;
  pOld->samples = 0;
  pbwtDestroy(pOld);
  free(yseq);
  pbwtCursorDestroy(uOld);
  pbwtCursorDestroy(uNew);
  return pNew;
}

PBWT * pbwtMerge2(PBWT * p, PBWT * q) {
  /*Assumes same sites in pbwts*/
  if (p->N != q->N) die("Cannot merge, PBWTs do not have the same sites");
  int nhaps = p->M + q->M;
  PBWT * out_pbwt = pbwtCreate(nhaps, 0);
  uchar * yseq = myalloc(nhaps, uchar);
  out_pbwt->yz = arrayCreate (1<<20, uchar);
  out_pbwt->sites = arrayCreate(p->N, Site);
  out_pbwt->chrom = strdup(p->chrom);

  PbwtCursor * cursor = pbwtCursorCreate(out_pbwt, TRUE, TRUE);
  PbwtCursor * pc = pbwtCursorCreate(p, TRUE, TRUE);
  PbwtCursor * qc = pbwtCursorCreate(q, TRUE, TRUE);

  int j, s;
  out_pbwt->samples = arrayCreate(nhaps, int);
  for (j = 0 ; j < p->M ; ++j)
    array(out_pbwt->samples, j, int) = arr(p->samples, j, int);
  for (j = p->M ; j < nhaps ; ++j)
    array(out_pbwt->samples, j, int) = arr(q->samples, j - p->M, int);

  for (s = 0; s < p->N; s++) {
    array(out_pbwt->sites, out_pbwt->N++, Site) = * arrp(p->sites, s, Site);
    // read and merge
    for (j = 0; j < p->M; j++) yseq[pc->a[j]] = pc->y[j];
    for (j = 0; j < q->M; j++) yseq[p->M + qc->a[j]] = qc->y[j];
    // pack merged haplotypes
    for (j = 0; j < nhaps; j++) cursor->y[j] = yseq[cursor->a[j]];
    pbwtCursorForwardsRead(pc);
    pbwtCursorForwardsRead(qc);
    pbwtCursorWriteForwards(cursor);
  }
  pbwtCursorToAFend(cursor, out_pbwt);
  free(yseq);
  pbwtCursorDestroy(cursor);
  pbwtCursorDestroy(pc);
  pbwtCursorDestroy(qc);

  return out_pbwt;
}

/******************* end of file *******************/
