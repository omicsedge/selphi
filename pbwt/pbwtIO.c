/*  File: pbwtIO.c
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
 * Description: read/write functions for pbwt package
 *
 * This is a slimmed down version of the original pbwt repository 
 * with added functionality for getting all matches of a certain length
 * between target haplotypes and all haplotypes in a reference panel.
 *-------------------------------------------------------------------
 */

#include "pbwt.h"
#include <ctype.h>

int nCheckPoint = 0 ;	/* if set non-zero write pbwt and sites files every n sites when parsing external files */

static BOOL isWriteImputeRef = FALSE ;	/* modifies WriteSites() and WriteHaplotypes() for pbwtWriteImputeRef */

/* basic function to store packed PBWT */

static void pbwtWrite (PBWT *p, FILE *fp) /* just writes compressed pbwt in yz */
{
  if (!p || !p->yz) die ("pbwtWrite called without a valid pbwt") ;
  if (!p->aFstart || !p->aFend) die ("pbwtWrite called without start and end indexes") ;
  /* version 2 added start and end indexes */
  if (fwrite ("PBW3", 1, 4, fp) != 4) /* version 3 with 8 byte pbwt size */
    die ("error writing PBWT in pbwtWrite") ;
  if (fwrite (&p->M, sizeof(int), 1, fp) != 1)
    die ("error writing M in pbwtWrite") ;
  if (fwrite (&p->N, sizeof(int), 1, fp) != 1)
    die ("error writing N in pbwtWrite") ;
  if (fwrite (p->aFstart, sizeof(int), p->M, fp) != p->M)
    die ("error writing aFstart in pbwtWrite") ;
  if (fwrite (p->aFend, sizeof(int), p->M, fp) != p->M)
    die ("error writing aFend in pbwtWrite") ;
  long n = arrayMax(p->yz) ;
  if (fwrite (&n, sizeof(long), 1, fp) != 1)
    die ("error writing n in pbwtWrite") ;
  if (fwrite ("    ", 1, 4, fp) != 4)
    die ("error writing padding space in pbwtWrite") ;
  if (fwrite (arrp(p->yz, 0, uchar), sizeof(uchar), arrayMax(p->yz), fp) != arrayMax(p->yz))
    die ("error writing data in pbwtWrite") ;

  fprintf(logFile, "\n [pbwt]: Saved %d haplotypes and %d sites to file\n", p->M, p->N);
}

static void pbwtWriteSites (PBWT *p, FILE *fp) {
  if (!p || !p->sites) die ("pbwtWriteSites called without sites") ;

  int i ;
  for (i = 0 ; i < p->N ; ++i)
    { Site *s = arrp(p->sites, i, Site) ;
      if (isWriteImputeRef)
	fprintf (fp, "site%d\t%d", i+1, s->x) ;
      else
	fprintf (fp, "%s\t%d", p->chrom ? p->chrom : ".", s->x) ;
      fprintf (fp, "\t%s", dictName (variationDict, s->varD)) ;
      fputc ('\n', fp) ;
    }
  if (ferror (fp)) die ("error writing sites file") ;
}

static void pbwtWriteSamples (PBWT *p, FILE *fp) {
  if (!p || !p->samples) die ("pbwtWriteSamples called without samples") ;

  int i ;
  for (i = 0 ; i < p->M ; i += 2) /* assume diploid for now */
    { Sample *s = sample (p, i) ;
      fprintf (fp, "%s", sampleName(s)) ;
      if (s->popD) fprintf (fp, "\tPOP:%s", popName(s)) ;
      if (s->mother) fprintf (fp, "\tMOTHER:%s", sampleName(sample (p, s->mother))) ;
      if (s->father) fprintf (fp, "\tFATHER:%s", sampleName(sample (p, s->father))) ;
      fputc ('\n', fp) ;
    }     
  if (ferror (fp)) die ("error writing samples file") ;
}

static void writeDataOffset (FILE *fp, char *name, Array data, Array offset, int N) {
  if (!offset || !data) die ("write %s called without data", name) ;
  int dummy = -1 ;	/* ugly hack to mark that we now use longs not ints */
  if (fwrite (&dummy, sizeof(int), 1, fp) != 1)
    die ("error writing marker in write %s", name) ;
  long n = arrayMax(data) ;
  if (fwrite (&n, sizeof(long), 1, fp) != 1)
    die ("error writing n in write %s", name) ;
  if (fwrite (arrp(data, 0, uchar), sizeof(uchar), n, fp) != n)
    die ("error writing data in write %s", name) ;
  if (fwrite (arrp(offset, 0, long), sizeof(long), N, fp) != N)
    die ("error writing offsets in write %s", name) ;
}

static void pbwtWriteMissing (PBWT *p, FILE *fp)
{ writeDataOffset (fp, "missing", p->zMissing, p->missingOffset, p->N) ; }

static void pbwtWriteReverse (PBWT *p, FILE *fp) {
  if (!p || !p->zz) die ("pbwtWriteReverse called without reverse pbwt") ;

  Array tz = p->yz ; p->yz = p->zz ;
  int* tstart = p->aFstart ; p->aFstart = p->aRstart ;
  int* tend = p->aFend ; p->aFend = p->aRend ;

  fprintf (logFile, "reverse: ") ; pbwtWrite (p, fp) ;
  
  p->yz = tz ; p->aFstart = tstart ; p->aFend = tend ;
}

void pbwtWriteAll (PBWT *p, char *root) {
  FILE *fp ;
#define FOPEN_W(tag)  if (!(fp = fopenTag (root, tag, "w"))) die ("failed to open %s.%s", root, tag)
  FOPEN_W("pbwt") ; pbwtWrite (p, fp) ; fclose (fp) ;
  if (p->sites) { FOPEN_W("sites") ; pbwtWriteSites (p, fp) ; fclose (fp) ; }
  if (p->samples) { FOPEN_W("samples") ; pbwtWriteSamples (p, fp) ; fclose (fp) ; }
  if (p->missingOffset) { FOPEN_W("missing") ; pbwtWriteMissing (p, fp) ; fclose (fp) ; }
  if (p->zz) { FOPEN_W("reverse") ; pbwtWriteReverse (p, fp) ; fclose (fp) ; }
}

void pbwtCheckPoint(PbwtCursor * u, PBWT * p) {
  static BOOL isA = TRUE;
  char fileNameRoot[20];

  pbwtCursorToAFend(u, p);
  sprintf(fileNameRoot, "check_%c", isA ? 'A' : 'B');
  pbwtWriteAll(p, fileNameRoot);

  isA = !isA;
}

/*******************************/

static PBWT *pbwtRead (FILE *fp) {
  int m, n ;
  long nz ;
  PBWT *p ;
  static char tag[5] = "test" ;
  char pad[4] ;
  int version ;

  if (fread (tag, 1, 4, fp) != 4) die ("failed to read 4 char tag - is file readable?") ;
  if (!strcmp (tag, "PBW3")) version = 3 ; /* current version */
  else if (!strcmp (tag, "PBW2")) version = 2 ; /* with 4 byte count */
  else if (!strcmp (tag, "PBWT")) version = 1 ; /* without start, end indexes */
  else if (!strcmp (tag, "GBWT")) version = 0 ; /* earliest version */
  else die ("failed to recognise file type %s in pbwtRead - was it written by pbwt?", tag) ;

  if (fread (&m, sizeof(int), 1, fp) != 1) die ("error reading m in pbwtRead") ;
  if (fread (&n, sizeof(int), 1, fp) != 1) die ("error reading n in pbwtRead") ;
  p = pbwtCreate (m, n) ;
  if (version > 1)		/* read aFstart and aFend */
    { p->aFstart = myalloc (m, int) ;
      if (fread (p->aFstart, sizeof(int), m, fp) != m) die ("error reading aFstart in pbwtRead") ;
      p->aFend = myalloc (m, int) ;
      if (fread (p->aFend, sizeof(int), m, fp) != m) die ("error reading aFend in pbwtRead") ;
    }
  else				/* set aFstart to 0..M-1, leave aFend empty */
    { p->aFstart = myalloc (m, int) ;
      int i ; for (i = 0 ; i < m ; ++i) p->aFstart[i] = i ;
    }

  if (version <= 2)
    { if (fread (&n, sizeof(int), 1, fp) != 1) die ("error reading pbwt file") ;
      nz = n ;
    }
  else
    if (fread (&nz, sizeof(long), 1, fp) != 1 ||
	fread (pad, 1, 4, fp) != 4) die ("error reading pbwt file") ;

  p->yz = arrayCreate (nz, uchar) ;
  array(p->yz, nz-1, uchar) = 0 ; /* sets arrayMax */
  if (fread (arrp(p->yz, 0, uchar), sizeof(uchar), nz, fp) != nz)
    die ("error reading data in pbwt file") ;

  fprintf(logFile, " [pbwt]: Read file with %d haplotypes and %d sites\n", p->M, p->N);
  return p ;
}

static BOOL readMatchChrom (char **pChrom, FILE *fp) {
  char *newChrom = fgetword (fp) ;

  if (strcmp (newChrom, "."))
    { if (!*pChrom) 
	*pChrom = strdup (newChrom) ;
      else if (strcmp (newChrom, *pChrom)) 
	return FALSE ;
    }
  return TRUE ;
}

Array pbwtReadFilterFile(FILE *fp, int n) {
  char ch;
  int j = 0;
  Array filter = arrayCreate(n, int);
  while (!feof(fp)) {
    ch = fgetc(fp);
    if (feof(fp)) break;
    if (ch != '\n') {
      if (j == n) die("Filter file is too big for pbwt");
      if (ch == '1') array(filter, j++, int) = 1; else array(filter, j++, int) = 0;
    }
  }
  if (j != n) die("Filter file is too small for pbwt");
  return filter;
}

static Array pbwtReadSitesFile (FILE *fp, char **chrom) {
  char c ;
  Site *s ;
  int line = 1 ;
  Array varTextArray = arrayCreate (256, char) ;
  Array sites = arrayCreate (4096, Site) ;

  while (!feof(fp))
    if (readMatchChrom (chrom, fp))	/* if p->chrom then match, else set if not "." */
      { if (feof(fp)) break ;
	s = arrayp(sites, arrayMax(sites), Site) ;
	s->x = 0 ; while (isdigit(c = fgetc(fp))) s->x = s->x*10 + (c-'0') ;
	if (!feof(fp) && c != '\n')
	  { if (!isspace (c)) die ("bad position line %d in sites file", line) ;
	    while (isspace(c = fgetc(fp)) && c != '\n') ;
	    if (c == '\n') die ("bad end of line at line %d in sites file", line) ;
	    int i = 0 ; array(varTextArray, i++, char) = c ;
	    while ((c = fgetc(fp)) && c != '\n') 
	      array(varTextArray, i++, char) = c ;
	    array(varTextArray, i, char) = 0 ;
	    dictAdd (variationDict, arrayp(varTextArray,0,char), &s->varD) ;
	    while (c != '\n' && !feof(fp)) c = fgetc(fp) ;
	  }
	++line ;
      }
    else if (!feof(fp))
      die ("failed to match chromosome in sites file: line %d", line) ;

  if (ferror (fp)) die ("error reading sites file") ;
  arrayDestroy (varTextArray) ;
  return sites ;
}

static void pbwtReadSites (PBWT *p, FILE *fp) {
  if (!p) die ("pbwtReadSites called without a valid pbwt") ;

  p->sites = pbwtReadSitesFile (fp, &p->chrom) ;
  if (arrayMax(p->sites) != p->N)
    die ("sites file contains %ld sites not %d as in pbwt", arrayMax(p->sites), p->N) ;
}


Array pbwtReadSamplesFile (FILE *fp) /* for now assume all samples diploid */
/* should add code to this to read father and mother and population
   propose to use IMPUTE2 format for this */
{
  char *name, c ;
  int line = 0 ;
  Array nameArray = arrayCreate (1024, char) ;
  Array samples = arrayCreate (1024, int) ;

  while (!feof(fp))
    { int n = 0 ;
      while ((c = getc(fp)) && !isspace(c) && !feof (fp)) array(nameArray, n++, char) = c ;
      if (feof(fp)) break ;
      if (!n) die ("no name line %ld in samples file", arrayMax(samples)+1) ;
      array(nameArray, n++, char) = 0 ;
      /* next bit of code deals with header lines for IMPUTE2 samples file, so can read that */
      if (!strcmp (arrp(nameArray,0,char), "ID_1") && !arrayMax(samples))
	{ while ((c = getc(fp)) && c != '\n' && !feof(fp)) ; /* remove header line */
	  while ((c = getc(fp)) && c != '\n' && !feof(fp)) ; /* and next line of zeroes?? */
	  continue ;
	}
      array(samples,arrayMax(samples),int) = sampleAdd (arrp(nameArray,0,char), 0, 0, 0) ;
      /* now remove the rest of the line, for now */
      while (c != '\n' && !feof(fp)) c = getc(fp) ;
    }
  arrayDestroy (nameArray) ;
  return samples ;
}

static void pbwtReadSamples (PBWT *p, FILE *fp) {
  if (!p) die ("pbwtReadSamples called without a valid pbwt") ;
  Array samples = pbwtReadSamplesFile (fp) ;
  if (arrayMax(samples) != p->M/2) 
    die ("wrong number of diploid samples: %d needed", p->M/2) ;
  p->samples = arrayReCreate(p->samples, p->M, int) ;
  int i ; 
  for (i = 0 ; i < arrayMax(samples) ; ++i)
    { array(p->samples, 2*i, int) = arr(samples, i, int) ;
      array(p->samples, 2*i+1, int) = arr(samples, i, int) ;
    }
  arrayDestroy (samples) ;
}

static void readDataOffset (FILE *fp, char *name, Array *data, Array *offset, int N) {
  long n ;			/* size of data file */
  int dummy ; 
  if (fread (&dummy, sizeof(int), 1, fp) != 1) 
    die ("read error in read %s", name) ;
  if (dummy != -1) n = dummy ;	/* old version with ints not longs */
  else if (fread (&n, sizeof(long), 1, fp) != 1) 
    die ("read error in read %s", name) ;

  *data = arrayReCreate (*data, n, uchar) ;
  if (fread (arrp(*data, 0, uchar), sizeof(uchar), n, fp) != n)
    die ("error reading z%s in pbwtRead%s", name, name) ;
  arrayMax(*data) = n ;
  fprintf (logFile, "read %ld chars compressed %s data\n", n, name) ;

  *offset = arrayReCreate (*offset, N, long) ;
  if (dummy != -1)		/* old version with ints not longs */
    { /* abuse *offset to hold ints, then update in place */
      if (fread (arrp(*offset, 0, int), sizeof(int), N, fp) != N) 
	die ("error reading %s in pbwtRead%s", name, name) ;
      for (n = N ; n-- ;) 
	arr(*offset, n, long) = arr(*offset, n, int) ; /* !! */
    }
  else
    if (fread (arrp(*offset, 0, long), sizeof(long), N, fp) != N)
      die ("error reading %s in pbwtRead%s", name, name) ;
  arrayMax(*offset) = N ;
}

static void pbwtReadMissing (PBWT *p, FILE *fp)
{ readDataOffset (fp, "missing", &p->zMissing, &p->missingOffset, p->N) ; }

static void pbwtReadReverse (PBWT *p, FILE *fp) {
  if (!p) die ("pbwtReadReverse called without a valid pbwt") ;

  PBWT *q = pbwtRead (fp) ;
  if (q->M != p->M || q->N != p->N)
    die ("M %d or N %d in reverse don't match %, %d in forward", q->M, q->N, p->M, p->N) ;
  p->zz = q->yz ; q->yz = 0 ;
  p->aRstart = q->aFstart ; q->aFstart = 0 ;
  p->aRend = q->aFend ; q->aFend = 0 ;
  pbwtDestroy (q) ;
}

PBWT *pbwtReadAll (char *root) {
  PBWT *p ;
  FILE *fp ;
  if ((fp = fopenTag (root, "pbwt", "r"))) { p = pbwtRead (fp) ; fclose (fp) ; } 
  else die ("failed to open %s.pbwt", root) ;
  if ((fp = fopenTag (root, "sites","r")))  { pbwtReadSites (p, fp) ; fclose (fp) ; }
  if ((fp = fopenTag (root, "samples","r"))) { pbwtReadSamples (p, fp) ; fclose (fp) ; }
  if ((fp = fopenTag (root, "missing","r"))) { pbwtReadMissing (p, fp) ; fclose (fp) ; }
  if ((fp = fopenTag (root, "reverse","r"))) { pbwtReadReverse (p, fp) ; fclose (fp) ; }
  return p ;
}

/******************* end of file *******************/
