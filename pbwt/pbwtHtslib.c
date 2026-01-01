/*  File: pbwtHtslib.c
 *  Original code authored by: Richard Durbin (rd@sanger.ac.uk)
 *  Copyright (C) Genome Research Limited, 2013
 *-------------------------------------------------------------------
 * Description: all the pbwt stuff that uses htslib, e.g. reading/writing vcf or bcf files
 *
 * This is a slimmed down version of the original pbwt repository 
 * with added functionality for getting all matches of a certain length
 * between target haplotypes and all haplotypes in a reference panel.
 *-------------------------------------------------------------------
 */

#include "utils.h"
#include "pbwt.h"
#include <htslib/synced_bcf_reader.h>
#include <htslib/faidx.h>
#include <ctype.h>		/* for toupper() */

const char *pbwtHtslibVersionString(void) {
    return hts_version();
}

static void readVcfSamples (PBWT *p, bcf_hdr_t *hr) {
  int i, k ;

  p->samples = arrayCreate (p->M, int) ;
  for (i = 0 ; i < p->M/2 ; ++i)
    { int k = sampleAdd (hr->samples[i],0,0,0) ;
      array(p->samples, 2*i, int) = k ; /* assume diploid - could be cleverer */
      array(p->samples, 2*i+1, int) = k ;
    }
}

static int variation (PBWT *p, const char *ref, const char *alt) {
  static char *buf = 0 ;
  static int buflen = 0 ;
  if (!buf) { buflen = 64 ; buf = myalloc (buflen, char) ; }
  int var ;
  if (strlen (ref) + strlen (alt) + 2 > buflen) 
    { do buflen *= 2 ; while (strlen (ref) + strlen (alt) + 2 > buflen) ;
      free (buf) ; buf = myalloc (buflen, char) ;
    }
  sprintf (buf, "%s\t%s", ref, alt) ;
  dictAdd (variationDict, buf, &var) ;
  return var ;
}

PBWT *pbwtReadVcfGT (char *filename) {
  int i, j ;

  bcf_srs_t *sr = bcf_sr_init();
  if (!bcf_sr_add_reader (sr, filename)) die ("failed to open good vcf file\n") ;

  bcf_hdr_t *hr = sr->readers[0].header;
  PBWT *p = pbwtCreate(bcf_hdr_nsamples(hr)*2, 0); /* assume diploid! */
  readVcfSamples(p, hr);
  p->sites = arrayCreate(10000, Site);
  PbwtCursor *u = pbwtCursorCreate(p, TRUE, TRUE);
  uchar *x = myalloc(p->M, uchar);

  uchar *xMissing = myalloc(p->M+1, uchar) ;
  xMissing[p->M] = Y_SENTINEL;  /* needed for efficient packing */
  long nMissing = 0;
  int nMissingSites = 0; 

  int mgt_arr = 0, *gt_arr = NULL;
  while (bcf_sr_next_line (sr)) 
    { bcf1_t *line = bcf_sr_get_line(sr,0) ;
      const char* chrom = bcf_seqname(hr,line) ;
      if (!p->chrom) p->chrom = strdup (chrom) ;
      else if (strcmp (chrom, p->chrom)) break ;
      int pos = line->pos + 1 ;       // bcf coordinates are 0-based
      char *ref, *REF; 
      ref = REF = strdup(line->d.allele[0]);
      while ( (*ref = toupper(*ref)) ) ++ref ;

      // get a copy of GTs
      int ngt = bcf_get_genotypes(hr, line, &gt_arr, &mgt_arr) ;
      if (ngt <= 0) continue ;  // it seems that -1 is used if GT is not in the FORMAT
      if (ngt != p->M && p->M != 2*ngt) die ("%d != %d GT values at %s:%d - not haploid or diploid?", 
          ngt, p->M, chrom, pos) ;

      memset (xMissing, 0, p->M) ;
      long wasMissing = nMissing ;
      /* copy the genotypes into array x[] */
      if (p->M == 2*ngt) // all GTs haploid: treat haploid genotypes as diploid homozygous A/A
        {
          for (i = 0 ; i < ngt ; i++)
            { if (gt_arr[i] == bcf_gt_missing)
                { x[2*i] = 0 ;
                  x[2*i+1] = 0; /* use ref for now */
                  xMissing[2*i] = 1 ;
                  xMissing[2*i+1] = 1;
                  nMissing+=2 ;
                }
              else {
                x[2*i] = bcf_gt_allele(gt_arr[i]) ;  // convert from BCF binary to 0 or 1
                x[2*i+1] = x[2*i] ;  // convert from BCF binary to 0 or 1
              }
            }
        }
      else
        {
          for (i = 0 ; i < p->M ; i++)
            { if (gt_arr[i] == bcf_int32_vector_end) 
                die ("unexpected end of genotype vector in VCF") ;
              if (gt_arr[i] == bcf_gt_missing)
                { x[i] = 0 ; /* use ref for now */
                  xMissing[i] = 1 ;
                  ++nMissing ;
                }
              else 
                x[i] = bcf_gt_allele(gt_arr[i]) ;  // convert from BCF binary to 0 or 1
            }
        }

      BOOL no_alt = line->n_allele == 1;
      int n_allele = no_alt ? 2 : line->n_allele;

      /* split into biallelic sites filling in as REF ALT alleles */
      /* not in the REF/ALT site */
      for (i = 1 ; i < n_allele ; i++)
        {
          char *alt, *ALT; 
          alt = ALT = no_alt ? "." : strdup(line->d.allele[i]);
          if (!no_alt) while ( (*alt = toupper(*alt)) ) ++alt;

          /* and pack them into the PBWT */
          for (j = 0 ; j < p->M ; ++j) u->y[j] = x[u->a[j]] == i ? 1 : 0;
          pbwtCursorWriteForwards(u);

          /* store missing information, if there was any */
          if (nMissing > wasMissing)
            { if (!wasMissing)
                { p->zMissing = arrayCreate(10000, uchar);
                  array(p->zMissing, 0, uchar) = 0; /* needed so missing[] has offset > 0 */
                  p->missingOffset = arrayCreate(1024, long);
                }
              array(p->missingOffset, p->N, long) = arrayMax(p->zMissing) ;
              pack3arrayAdd (xMissing, p->M, p->zMissing); /* NB original order, not pbwt sort */
              nMissingSites++ ;
            }
          else if (nMissing)
            array(p->missingOffset, p->N, long) = 0;

          // add the site
          Site *s = arrayp(p->sites, p->N++, Site);
          s->x = pos;
          s->varD = variation(p, REF, ALT);
          
          /* free ALT string if it was allocated */
          if (!no_alt && ALT) { free(ALT); ALT = NULL; }
        }

      /* free REF string after processing all alleles for this line */
      if (REF) { free(REF); REF = NULL; }

      if (nCheckPoint && !(p->N % nCheckPoint))  pbwtCheckPoint (u, p) ;
    }
  pbwtCursorToAFend (u, p) ;

  if (gt_arr) free (gt_arr) ;
  bcf_sr_destroy (sr) ;
  free (x) ; pbwtCursorDestroy (u) ;  
  free (xMissing) ;

  fprintf(logFile, " [pbwt]: Read genotypes from %s with %ld haplotypes and %ld sites on chromosome %s\n", 
         filename, p->M, p->N, p->chrom);
  if (p->missingOffset) fprintf (logFile, " [pbwt]: %ld missing values at %d sites\n", 
         nMissing, nMissingSites) ;

  return p ;
}

/******* end of file ********/
