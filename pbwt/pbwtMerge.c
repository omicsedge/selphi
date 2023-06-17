/*
 * Based on code by Richard Durbin
 * Modified for merging 2 pbwt objects in memory
 */
#include "pbwt.h"

#include <string.h>
#include <errno.h>

// Synced reading of multiple PBWTs. 
// Note: all operations in-memory for now, buffered input in future?
typedef struct
{
	// all operations in-memory for now, buffered input in future?
	int n;				 // number of PBWTs
	PBWT **pbwt;
	int *cpos;		 // current position of each PBWT (site index)
	int mpos;			 // minimum position across all PBWT (position)
	char *mals;		 // alleles at the mpos (multiple records with different alleles can be present)
	PbwtCursor **cursor;
	int *unpacked;
}
pbwt_reader_t;

// Return value: 0 if all PBWTs finished, otherwise the current position is returned
static int pbwt_reader_next(pbwt_reader_t *reader, int nshared)
{
	int i, min_pos = INT_MAX;
	char *min_als  = NULL;

	// advance all readers, first looking at coordinates only
	for (i=0; i<reader->n; i++)
	{  
		PBWT *p = reader->pbwt[i];
		int j		= reader->cpos[i];
		if ( j>=p->N ) continue;		// no more sites in this pbwt

		Site *site = arrp(p->sites, j, Site);
		char *als  = dictName(variationDict, site->varD);

		// assuming:
		//	- one chromosome only (no checking sequence name)
		//	- sorted alleles (strcmp() on als)
		while ( j < p->N && site->x <= reader->mpos && (!reader->mals || strcmp(als,reader->mals)<=0) )
		{
			site = arrp(p->sites, j, Site);
			als  = dictName(variationDict, site->varD);
			reader->cpos[i] = j++;
		}
		if ( reader->cpos[i]+1 >= p->N && site->x == reader->mpos && (!reader->mals || !strcmp(als,reader->mals)) )
		{
			// this pbwt is positioned on the last site which has been read before
			reader->cpos[i] = p->N;
			continue;
		}

		if ( reader->cpos[i] < p->N && site->x < min_pos )
		{
			min_pos = site->x;
			min_als = als;
		}
		if ( site->x==min_pos && (!min_als || strcmp(als,min_als)<0) ) min_als = als;
	}
	if ( min_pos==INT_MAX )
	{
		reader->mpos = 0;
		reader->mals = NULL;
	}
	else
	{
		reader->mpos = min_pos;
		reader->mals = min_als;
	}
	return reader->mpos;
}

static pbwt_reader_t * pbwt_read2(PBWT * p, PBWT * q){
  pbwt_reader_t * reader = calloc(1, sizeof(pbwt_reader_t));
  reader->n = 2;
  reader->pbwt = myalloc(2, PBWT*);
  reader->cpos = myalloc(2, int);
  reader->cursor = myalloc(2, PbwtCursor*);
  reader->unpacked = myalloc(2, int);

  reader->pbwt[0] = p;
  reader->cursor[0] = pbwtNakedCursorCreate(p->M, 0);
  reader->pbwt[1] = q;
  reader->cursor[1] = pbwtNakedCursorCreate(q->M, 0);
  for (int i = 0; i < 2; i++) {
    reader->cpos[i] = 0;
    reader->unpacked[i] = 0;
  }
  return reader;
}

static void pbwt_read2_destroy(pbwt_reader_t *reader)
{
  int i;
  for (i = 0; i < reader->n; i++) pbwtCursorDestroy(reader->cursor[i]);
  free(reader->pbwt);
  free(reader->cursor);
  free(reader->unpacked);
  free(reader->cpos);
  free(reader);
}

PBWT * pbwtMerge2(PBWT * p, PBWT * q) {
  pbwt_reader_t *reader = pbwt_read2(p, q);
  int nhaps = p->M + q->M;
  PBWT * out_pbwt = pbwtCreate(nhaps, 0);
  PbwtCursor *cursor = pbwtNakedCursorCreate(nhaps, 0);
  uchar *yseq = myalloc(nhaps, uchar);
  out_pbwt->yz = arrayCreate (1<<20, uchar);
  out_pbwt->sites = arrayReCreate(out_pbwt->sites, p->N, Site);
  out_pbwt->chrom = strdup(p->chrom);

  int pos, i, j;
  out_pbwt->samples = arrayCreate(nhaps, int);
  for (j = 0 ; j < p->M ; ++j)
    array(out_pbwt->samples, j, int) = arr(p->samples, j, int);
  for (j = p->M ; j < nhaps ; ++j)
    array(out_pbwt->samples, j, int) = arr(q->samples, j - p->M, int);

  int nfiles = 2;
  while ( (pos=pbwt_reader_next(reader, nfiles)) ) {
    // Merge only records shared by all files
    for (i=0; i < nfiles; i++) {
      PBWT *p_ = reader->pbwt[i];
      Site *site = arrp(p_->sites, reader->cpos[i], Site);
      // Both position and alleles must match. This requires that the records are sorted by alleles.
      if ( site->x != pos ) break;
      char *als = dictName(variationDict, site->varD);
      if ( strcmp(als,reader->mals) ) break;
    }
    if ( i != nfiles ) {
      // intersection: skip records which are not present in all files
      for (i = 0; i < nfiles; i++) {
        PBWT * p_ = reader->pbwt[i];
        Site *site = arrp(p_->sites, reader->cpos[i], Site);
        if ( site->x != pos ) continue;
        char *als = dictName(variationDict, site->varD);
        if ( strcmp(als, reader->mals) ) continue;
        PbwtCursor * c = reader->cursor[i];
        reader->unpacked[i] += unpack3(arrp(p_->yz, reader->unpacked[i], uchar), p_->M, c->y, 0);
        pbwtCursorForwardsA(c);
      }
      continue;
    }
    // read and merge
    int ihap = 0;
    for (i=0; i<nfiles; i++) {
      PbwtCursor * c = reader->cursor[i];
      PBWT * p_ = reader->pbwt[i];
      Site *site = arrp(p_->sites, reader->cpos[i], Site);
      reader->unpacked[i] += unpack3(arrp(p_->yz, reader->unpacked[i], uchar), p_->M, c->y, 0);
      for (j=0; j < p_->M; j++) yseq[ihap + c->a[j]] = c->y[j];
      pbwtCursorForwardsA(c);
      ihap += p_->M;
    }
    // pack merged haplotypes
    for (j=0; j<nhaps; j++) cursor->y[j] = yseq[cursor->a[j]];
    pack3arrayAdd(cursor->y, out_pbwt->M, out_pbwt->yz);
    pbwtCursorForwardsA(cursor);
    // insert new site
    arrayExtend(out_pbwt->sites, out_pbwt->N + 1);
    Site * site = arrayp(out_pbwt->sites, out_pbwt->N, Site);
    site->x = pos;
    dictAdd(variationDict, reader->mals, &site->varD);
    out_pbwt->N++;
  }
  pbwtCursorToAFend(cursor, out_pbwt);

  free(yseq);
  pbwtCursorDestroy(cursor);
  pbwt_read2_destroy(reader);
  return out_pbwt;
}
