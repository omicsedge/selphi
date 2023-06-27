/*  File: utils.c
 *  Author: Richard Durbin (rd@sanger.ac.uk)
 *  Modified by Daniel Lawson (dan.lawson@bristol.ac.uk) in December 2014, adding gzip output for paintSparse
 *  Copyright (C) Genome Research Limited, 1996-
 *-------------------------------------------------------------------
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free Software
 * Foundation; either version 2.1 of the License, or (at your option) any later
 * version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 * You should have received a copy of the GNU Lesser General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 *-------------------------------------------------------------------
 * Description: core utility functions
 * Exported functions:
 * HISTORY:
 * Last edited: Dec 28 14:02 2014 (dl)
 * adding gzip output for paintSparse
 * Created: Thu Aug 15 18:32:26 1996 (rd)
 *-------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>
#include "utils.h"

void die (char *format, ...)
{
  va_list args ;

  va_start (args, format) ;
  fprintf (stderr, "FATAL ERROR: ") ;
  vfprintf (stderr, format, args) ;
  fprintf (stderr, "\n") ;
  va_end (args) ;

  exit (-1) ;
}

void warn (char *format, ...)
{
  static int count = 0 ; 
  va_list args ;

  va_start (args, format) ;
  fprintf (stderr, "ERROR: ") ;
  vfprintf (stderr, format, args) ;
  fprintf (stderr, "\n") ;
  va_end (args) ;

  if (++count > 9) die ("too many errors") ;
}

long int totalAllocated = 0 ;

void *_myalloc (long size)
{
  void *p = (void*) malloc (size) ;
  if (!p) die ("myalloc failure requesting %d bytes", size) ;
  totalAllocated += size ;
  return p ;
}

void *_mycalloc (long number, int size)
{
  void *p = (void*) calloc (number, size) ;
  if (!p) die ("mycalloc failure requesting %d of size %d bytes", number, size) ;
  totalAllocated += number*size ;
  return p ;
}

/*************************************************/

FILE *fopenTag (char* root, char* tag, char* mode)
{
  if (strlen (tag) > 30) die ("tag %s in fopenTag too long - should be < 30 chars", tag) ;
  char *fileName = myalloc (strlen (root) + 32, char) ;
  strcpy (fileName, root) ;
  strcat (fileName, ".") ;
  strcat (fileName, tag) ;
  FILE *f = fopen (fileName, mode) ;
  free (fileName) ;
  return f ;
}

gzFile gzopenTag (char* root, char* tag, char* mode)
{
  if (strlen (tag) > 40) die ("tag %s in gzopenTag too long - should be < 30 chars", tag) ;
  char *fileName = myalloc (strlen (root) + 42, char) ;
  strcpy (fileName, root) ;
  strcat (fileName, ".") ;
  strcat (fileName, tag) ;
  gzFile f = gzopen (fileName, mode) ;
  free (fileName) ;
  return f ;
}

/*************************************************/

char *fgetword (FILE *f)	// pass NULL to free alloced memory
{
  int n = 0 ;
  static char *buf = 0 ;
  int bufSize = 64 ;
  char *cp ;

  if (!f) { if (buf) free(buf); buf = NULL; return NULL; }

  if (!buf) buf = myalloc (bufSize, char) ;
  cp = buf ;
  while (!feof (f) && (*cp = getc (f)))
    if (isgraph(*cp) && !isspace(*cp))
      { ++cp ; ++n ;
	if (n >= bufSize)
	  { bufSize *= 2 ;
	    if (!(buf = (char*) realloc (buf, bufSize)))
	      die ("fgetword realloc failure requesting %d bytes", bufSize) ;
	  }
      }
    else
      { while ((isspace(*cp) || !isgraph(*cp)) && *cp != '\n' && !feof(f)) *cp = getc (f) ;
	ungetc (*cp, f) ;
	break ;
      }
  *cp = 0 ;
  return buf ;
}

/*************************************************/
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 50

void printProgress(double percentage) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r\t%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

/********************* end of file ***********************/
