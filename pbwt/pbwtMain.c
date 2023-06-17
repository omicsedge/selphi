/*  File: pbwtMain.c
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
 * This is a slimmed down version of the original pbwt repository 
 * with added functionality for getting all matches of a certain length
 * between target haplotypes and all haplotypes in a reference panel.
 *-------------------------------------------------------------------
 */

#include "pbwt.h"
#include "version.h"

#include <math.h>


/*********************************************************/

char * commandLine = "";

static void recordCommandLine(int argc, char * argv[]) {
  if (!argc) return;

  int i, len = 0;
  for (i = 0; i < argc; ++i) len += (1 + strlen(argv[i]));
  commandLine = myalloc(len, char);
  strcpy(commandLine, argv[0]);
  for (i = 1; i < argc; ++i) {
    strcat(commandLine, " ");
    strcat(commandLine, argv[i]);
  }
}

/*********************************************************/

/* a couple of utilities for opening/closing files */
#define FOPEN(name, mode) if (!strcmp(argv[1], "-")) fp = !strcmp(mode, "r") ? stdin : stdout; \
  else if (!(fp = fopen(argv[1], mode))) die("failed to open %s file %s", name, argv[1])
#define FCLOSE if (strcmp(argv[1], "-")) fclose(fp)
#define LOPEN(name, mode) if (!strcmp(argv[2], "-")) lp = !strcmp(mode, "r") ? stdin : stdout; \
  else if (!(lp = fopen(argv[2], mode))) die("failed to open %s file", name, argv[2])
#define LCLOSE if (strcmp(argv[2], "-")) fclose(lp)
#define LOGOPEN(name) if (!strcmp(argv[1], "-")) logFile = stderr; \
  else if (!(logFile = fopen(argv[1], "w"))) die("failed to open %s file %s", name, argv[1])
#define LOGCLOSE if (logFile && !(logFile == stderr)) fclose(logFile)

const char * pbwtCommitHash(void) {
  return PBWT_COMMIT_HASH;
}

FILE * logFile; /* log file pointer */

int main(int argc, char * argv[]) {
  FILE * fp;
  FILE * lp;
  PBWT * p = 0;
  Array test;

  logFile = stderr;

  pbwtInit();

  --argc;
  ++argv;
  recordCommandLine(argc, argv);

  if (!argc) /* print help */ {
    fprintf(stderr, "Program: pbwt\n");
    fprintf(stderr, "Version: %d.%d%s%s (using htslib %s)\n", pbwtMajorVersion, pbwtMinorVersion,
      strcmp(pbwtCommitHash(), "") == 0 ? "" : "-", pbwtCommitHash(), pbwtHtslibVersionString());
    fprintf(stderr, "Forked from: Richard Durbin [rd@sanger.ac.uk]\n");
    fprintf(stderr, "Modified and maintained by SelfDecode\n");
    fprintf(stderr, "Usage: pbwt [ -<command> [options]* ]+\n");
    fprintf(stderr, "Commands:\n");
    fprintf(stderr, "  -log <file>               log file; '-' for stderr\n");
    fprintf(stderr, "  -check                    do various checks\n");
    fprintf(stderr, "  -stats                    print stats depending on commands; writes to stdout\n");
    fprintf(stderr, "  -readAll <rootname>       read .pbwt and if present .sites, .samples, .missing\n");
    fprintf(stderr, "  -readVcfGT <file>         read GTs from vcf or bcf file; '-' for stdin vcf only ; biallelic sites only - require diploid!\n");
    fprintf(stderr, "  -checkpoint <n>           checkpoint every n sites while reading\n");
    fprintf(stderr, "  -writeSites <file>        write sites file; '-' for stdout\n");
    fprintf(stderr, "  -writeAll <rootname>      write .pbwt and if present .sites, .samples, .missing\n");
    fprintf(stderr, "  -selectSites <file>       select sites as in sites file\n");
    fprintf(stderr, "  -selectSamples <file>     select samples as in samples file\n");
    fprintf(stderr, "  -referenceMatch <root> <L> find matches in reference panel longer than L\n");
  }

  timeUpdate(logFile);
  while (argc) {
    if (!( ** argv == '-'))
      die("not well formed command %s\nType pbwt without arguments for help", * argv);
    else if (!strcmp(argv[0], "-check")) {
      isCheck = TRUE;
      argc -= 1;
      argv += 1;
    } else if (!strcmp(argv[0], "-stats")) {
      isStats = FALSE;
      argc -= 1;
      argv += 1;
    } else if (!strcmp(argv[0], "-log") && argc > 1) {
      LOGCLOSE;
      LOGOPEN("log");
      argc -= 2;
      argv += 2;
    } else if (!strcmp(argv[0], "-readAll") && argc > 1) {
      p = pbwtReadAll(argv[1]);
      argc -= 2;
      argv += 2;
    } else if (!strcmp(argv[0], "-readVcfGT") && argc > 1) {
      if (p) pbwtDestroy(p);
      p = pbwtReadVcfGT(argv[1]);
      argc -= 2;
      argv += 2;
    } else if (!strcmp(argv[0], "-writeSites") && argc > 1) {
      FOPEN("writeSites", "w");
      pbwtWriteSites(p, fp);
      FCLOSE;
      argc -= 2;
      argv += 2;
    } else if (!strcmp(argv[0], "-writeAll") && argc > 1) {
      pbwtWriteAll(p, argv[1]);
      argc -= 2;
      argv += 2;
    } else if (!strcmp(argv[0], "-checkpoint") && argc > 1) {
      nCheckPoint = atoi(argv[1]);
      argc -= 2;
      argv += 2;
    } else if (!strcmp(argv[0], "-selectSamples") && argc > 1) {
      FOPEN("selectSamples", "r");
      p = pbwtSelectSamples(p, fp);
      argc -= 2;
      argv += 2;
    } else if (!strcmp(argv[0], "-selectSites") && argc > 1) {
      FOPEN("selectSites", "r");
      char * chr = 0;
      Array sites = pbwtReadSitesFile(fp, & chr);
      if (strcmp(chr, p -> chrom)) die("chromosome mismatch in selectSites");
      p = pbwtSelectSites(p, sites, FALSE);
      free(chr);
      arrayDestroy(sites);
      argc -= 2;
      argv += 2;
    } else if (!strcmp(argv[0], "-referenceMatch") && argc > 2) {
      referenceMatch(p, argv[1], atoi(argv[2]));
      argc -= 3;
      argv += 3;
    } else
      die("unrecognised command %s\nType pbwt without arguments for help", * argv);
    timeUpdate(logFile);
  }
  if (p) pbwtDestroy(p);
  if (variationDict) dictDestroy(variationDict);
  sampleDestroy();
  fgetword(NULL); // to keep valgrind happy, free malloced memory
  LOGCLOSE;
  return 0;
}
/******************* end of file *******************/
