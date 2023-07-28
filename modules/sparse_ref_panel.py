"""
SparseReferencePanel is a class that stores reference genotypes
in chunked boolean sparse arrays. Chunking the arrays allows for
multi-threaded ingestion of genotype files. Loading the chunks
also provides faster loading times, lower memory usage, and faster
slicing times.

The reference panel is stored in a zstd-compressed zip archive:
    metadata - json file containing useful information about the sparse arrays
    variants - binary array of variants in panel
    chunks - binary array of chunks with start and stop positions
    haplotypes/* - chunks stored as boolean sparse npz files

Chunks are cached when they are read into memory. While a few seconds 
processing time is required to load each chunk, consecutive reads are very fast.

Files can only contain one chromosome, and variants are assumed to be sorted by position.

To open xsi as sparse reference:
ref_panel = SparseReferencePanel("/data/30x/chr20.srp").from_xsi("/data/30x/chr20.xsi", threads=4)

To open vcf/bcf as sparse reference:
ref_panel = SparseReferencePanel("/data/30x/chr20.srp").from_bcf("/data/30x/chr20.bcf", threads=4)

To open existing sparse reference file:
ref_panel = SparseReferencePanel("/data/30x/chr20.srp")

To get haplotype calls, the class acts like a sparse matrix:
sparse_matrix = ref_panel[variants, haplotypes]

If full sparse matrix functionality is needed, use all:
sparse_matrix = ref_panel.all

To select a part of the chromosome by position, use range:
sparse_matrix = ref_panel.range(start_pos, end_pos)
"""
import os
import subprocess
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Union
import json
from datetime import datetime
from zipfile import ZipFile
from tempfile import TemporaryDirectory

from zstd import compress, uncompress
import numpy as np
from scipy import sparse
from cachetools import LRUCache, cachedmethod

from joblib import Parallel, delayed

import cyvcf2

from .utils import tqdm_joblib


class SparseReferencePanel:
    """Class for working with ref panels stored as sparse matrix"""

    def __init__(self, filepath: str, cache_size: int = 2) -> None:
        self.filepath = filepath
        self.variant_dtypes = np.dtype(
            [("chr", "<U21"), ("pos", int), ("ref", "<U21"), ("alt", "<U21")]
        )
        if not os.path.exists(self.filepath):
            self._create()
        self.metadata = self._load_metadata()
        self.variants: np.ndarray = self._load_variants()
        self.chunks: np.ndarray = self._load_chunks()
        self.ids: List[str] = self._load_ids()
        self._cache = LRUCache(maxsize=cache_size)

    def __getitem__(self, key: Tuple[Union[int, list, slice]]) -> sparse.csc_matrix:
        """Get sparse matrix of boolean genotypes"""
        if not isinstance(key, tuple):
            raise TypeError("Both variant and haplotype slices must be provided")
        # handle single row
        if isinstance(key[0], int):
            return self._load_haplotypes(key[0] // self.chunk_size)[
                key[0] % self.chunk_size, key[1]
            ]
        # handle slice
        if isinstance(key[0], slice):
            chunk_step = -1 if key[0].step and key[0].step < 0 else 1
            if not key[0].start and key[0].stop is None:
                return sparse.vstack(
                    [
                        self._load_haplotypes(chunk)[:: key[0].step, key[1]]
                        for chunk in self.chunks[:, 0]
                    ]
                )
            row_stop = (
                min([key[0].stop, self.n_variants]) if key[0].stop else self.n_variants
            )
            chunks = list(
                range(
                    (key[0].start or 0) // self.chunk_size,
                    row_stop // self.chunk_size + 1,
                    chunk_step,
                )
            )
            if len(chunks) == 0:
                raise IndexError("No variants to return")

            if len(chunks) == 1:
                return self._load_haplotypes(chunks[0])[
                    key[0].start
                    % self.chunk_size : row_stop
                    % self.chunk_size : key[0].step,
                    key[1],
                ]

            slices = (
                [slice(key[0].start % self.chunk_size, None, key[0].step)]
                + [slice(None, None, key[0].step)] * (len(chunks) - 2)
                + [slice(None, row_stop % self.chunk_size, key[0].step)]
            )
            return sparse.vstack(
                [
                    self._load_haplotypes(chunk)[slice_, key[1]]
                    for chunk, slice_ in zip(chunks, slices)
                ]
            )

        # handle list of indexes
        if isinstance(key[0], list) and isinstance(key[0][0], int):
            rows = np.array(key[0])
        elif isinstance(key[0], np.ndarray):
            rows = key[0]
        else:
            raise TypeError("Variant selection must be int, List[int], or slice")

        chunks, splits = np.unique(rows // self.chunk_size, return_index=True)
        if any(chunk not in self.chunks[:, 0] for chunk in chunks):
            raise IndexError("Variants index out of range")

        chunk_idx = np.split(rows % self.chunk_size, splits[1:])

        return sparse.vstack(
            [
                self._load_haplotypes(chunk).tocsr()[idx, :]
                for chunk, idx in zip(chunks, chunk_idx)
            ]
        ).tocsc()[:, key[1]]

    def _create(self):
        """Create an empty file"""
        print("Creating new sparse matrix archive")
        if os.path.dirname(self.filepath):
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with ZipFile(self.filepath, mode="x") as archive:
            with archive.open("metadata", "w") as metadata:
                metadata.write(
                    compress(json.dumps({"created_at": str(datetime.now())}).encode())
                )
            with archive.open("variants", "w") as variants:
                variants.write(
                    compress(np.array([], dtype=self.variant_dtypes).tobytes())
                )
            with archive.open("chunks", "w") as chunks:
                chunks.write(compress(np.array([], dtype=int).tobytes()))

    def _save(self, hap_dir: str):
        """Update archive"""
        self.metadata["updated_at"] = str(datetime.now())
        with ZipFile(self.filepath, mode="w") as archive:
            with archive.open("metadata", "w") as metadata:
                metadata.write(compress(json.dumps(self.metadata).encode()))
            with archive.open("variants", "w") as variants:
                variants.write(compress(self.variants.tobytes()))
            with archive.open("IDs", "w") as ids:
                ids.write(compress("\n".join(self.ids).encode()))
            with archive.open("chunks", "w") as chunks:
                chunks.write(compress(self.chunks.tobytes()))
            for file in Path(hap_dir).iterdir():
                archive.write(file, arcname=os.path.join("haplotypes", file.name))

    def _load_metadata(self) -> dict:
        """Load metadata from archive"""
        with ZipFile(self.filepath, mode="r") as archive:
            with archive.open("metadata") as obj:
                return json.loads(uncompress(obj.read()))

    def _load_variants(self) -> np.ndarray:
        """Load variants from archive"""
        with ZipFile(self.filepath, mode="r") as archive:
            with archive.open("variants") as obj:
                return np.frombuffer(uncompress(obj.read()), dtype=self.variant_dtypes)

    def _load_ids(self) -> List[str]:
        """Load string formatted variant IDs from archive"""
        try:
            with ZipFile(self.filepath, mode="r") as archive:
                with archive.open("IDs") as obj:
                    return uncompress(obj.read()).decode().split("\n")
        except KeyError:
            return [
                "-".join([str(col) for col in variant]) for variant in self.variants
            ]

    def _ingest_variants(self, vcf_path: str):
        """Load variants from a vcf or bcf"""
        vcf_obj = cyvcf2.VCF(vcf_path)
        variants = [
            (variant.CHROM, variant.POS, variant.REF, variant.ALT[0])
            for variant in vcf_obj
        ]
        vcf_obj.close()
        # Store as numpy for array operations
        self.variants = np.fromiter(variants, dtype=self.variant_dtypes)
        # Also store as txt to avoid truncated indel alleles
        self.ids = ["-".join([str(col) for col in line]) for line in variants]

        chroms = np.unique([variant[0] for variant in self.variants])
        if chroms.size > 1:
            raise ValueError("Only one chromosome per file is supported")

        self.chunks = np.array(
            [
                [idx, chunk[0][1], chunk[-1][1]]
                for idx, chunk in enumerate(
                    np.split(
                        self.variants,
                        range(self.chunk_size, self.variants.size, self.chunk_size),
                    )
                )
            ],
            dtype=int,
        )
        self.metadata.update(
            {
                "chromosome": str(chroms[0]),
                "n_variants": int(self.variants.size),
                "min_position": int(self.variants[0][1]),
                "max_position": int(self.variants[-1][1]),
                "n_chunks": self.chunks.shape[0],
            }
        )

    def _load_chunks(self) -> np.ndarray:
        with ZipFile(self.filepath, mode="r") as archive:
            with archive.open("chunks") as obj:
                chunks_ = np.frombuffer(uncompress(obj.read()), dtype=int)
        if chunks_.size > 0:
            return np.reshape(chunks_, (self.n_chunks, 3))
        return chunks_

    @cachedmethod(lambda self: self._cache)
    def _load_haplotypes(self, chunk: int) -> sparse.csc_matrix:
        """Load a sparse matrix from archived npz"""
        with ZipFile(self.filepath, mode="r") as archive:
            with archive.open(f"haplotypes/{chunk}.npz") as obj:
                return sparse.load_npz(BytesIO(uncompress(obj.read())))

    def _std_out_to_sparse(self, command: str, chunk: int, tmpdir: str) -> tuple:
        """Convert std_out of command to sparse matrix"""
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
        # account for when the same position appears in multiple chunks
        offset = 0
        if self.chunks[chunk][1] == self.chunks[chunk - 1][2]:
            for variant in range(chunk * self.chunk_size - 1, 0, -1):
                if self.variants[variant][1] == self.chunks[chunk][1]:
                    offset += 1
                else:
                    break
        matrix = sparse.csc_matrix(
            np.loadtxt(BytesIO(result.stdout), delimiter="|", dtype=np.bool_)[
                offset : offset + self.chunk_size, :
            ]
        )
        # confirm correct number of variants
        assert matrix.shape[0] == self.chunk_size or (
            chunk == self.n_chunks - 1
            and matrix.shape[0] == self.n_variants % self.chunk_size
        )
        npz_ = BytesIO()
        sparse.save_npz(npz_, matrix)
        npz_.seek(0)
        with open(os.path.join(tmpdir, f"{chunk}.npz"), "wb") as fout:
            fout.write(compress(npz_.read()))
        return matrix.shape[1]

    def _ingest_haplotypes(self, commands: List[str], threads: int = 1):
        """Load haplotypes from non-npz file"""
        with TemporaryDirectory() as tmpdir:
            hap_dir = os.path.join(tmpdir, "haplotypes")
            os.makedirs(hap_dir)
            with tqdm_joblib(
                total=len(commands),
                desc="Saving haplotypes as sparse matrices",
                ncols=75,
                bar_format="{desc}:\t\t{percentage:3.0f}% in {elapsed}",
            ):
                haps = Parallel(n_jobs=threads,)(
                    delayed(self._std_out_to_sparse)(command, chunk, hap_dir)
                    for chunk, command in enumerate(commands)
                )
            hap_counts = list(set(haps))
            assert len(hap_counts) == 1

            self.metadata.update({"n_haps": int(hap_counts[0])})
            self._save(hap_dir)

    def _ingest_xsi_haplotypes(self, xsi_path: str, threads: int = 1):
        """Load haplotypes from xsi file"""
        # -r includes overlapping indels, but is faster than -t, so use both
        commands = [
            (
                f"xsqueezeit -x -f {xsi_path} -p "
                f'-r "{self.chromosome}:{chunk[1]}-{chunk[2]}" | '
                f"bcftools query -t {self.chromosome}:{chunk[1]}-{chunk[2]} "
                "-f '[|%GT]\n' | sed s'/|//'"
            )
            for chunk in self.chunks
        ]
        self._ingest_haplotypes(commands, threads)

    def from_xsi(
        self,
        xsi_path: str,
        chunk_size: int = 10 ** 4,
        threads: int = 1,
        replace_file: bool = False,
    ):
        """Convert an xsi file to sparse matrix"""
        if self.n_variants > 0 and not replace_file:
            print("Variants have already been loaded")
            return
        if not os.path.exists(xsi_path):
            raise FileNotFoundError(f"Missing input file: {xsi_path}")
        xsi_bcf = xsi_path + "_var.bcf"
        if not os.path.exists(xsi_bcf):
            raise FileNotFoundError(f"Missing input file: {xsi_bcf}")
        self.metadata["source_file"] = xsi_path
        self.metadata["chunk_size"] = chunk_size
        self._ingest_variants(xsi_bcf)
        self._ingest_xsi_haplotypes(xsi_path, threads)

        return self

    def _ingest_bcf_haplotypes(self, bcf_path: str, threads: int = 1):
        """Load haplotypes from vcf/bcf file"""
        # -r includes overlapping indels, but is faster than -t, so use both
        commands = [
            (
                f"bcftools view -r {self.chromosome}:{chunk[1]}-{chunk[2]} "
                f"{bcf_path} | bcftools query -f '[|%GT]\n' "
                f"-t {self.chromosome}:{chunk[1]}-{chunk[2]} | sed s'/|//'"
            )
            for chunk in self.chunks
        ]
        self._ingest_haplotypes(commands, threads)

    def from_bcf(
        self,
        bcf_path: str,
        chunk_size: int = 10 ** 4,
        threads: int = 1,
        replace_file: bool = False,
    ):
        """Convert a vcf/bcf file to sparse matrix"""
        if self.n_variants > 0 and not replace_file:
            print("Variants have already been loaded")
            return
        if not os.path.exists(bcf_path):
            raise FileNotFoundError(f"Missing input file: {bcf_path}")
        if not (os.path.exists(bcf_path + ".tbi") or os.path.exists(bcf_path + ".csi")):
            print(f"Indexing input file: {bcf_path}")
            subprocess.run(f"tabix {bcf_path}", shell=True)

        self.metadata["source_file"] = bcf_path
        self.metadata["chunk_size"] = chunk_size
        self._ingest_variants(bcf_path)
        self._ingest_bcf_haplotypes(bcf_path, threads)

        return self

    @property
    def n_variants(self) -> int:
        """Get number of variants"""
        return self.metadata.get("n_variants", 0)

    @property
    def n_haps(self) -> int:
        """Get number of haplotypes"""
        return self.metadata.get("n_haps", 0)

    @property
    def shape(self) -> int:
        """Get shape of full matrix"""
        return (self.n_variants, self.n_haps)

    @property
    def n_chunks(self) -> int:
        """Get number of chunks"""
        return self.metadata.get("n_chunks", 0)

    @property
    def chunk_size(self) -> int:
        """Number of variants per chunk"""
        return self.metadata.get("chunk_size", 0)

    @property
    def max_position(self) -> int:
        """Assumes only 1 chromosome and sorted variants"""
        return self.metadata.get("max_position", 0)

    @property
    def chromosome(self) -> str:
        """Get chromosome"""
        return self.metadata.get("chromosome", "")

    @property
    def empty(self) -> bool:
        return self.n_variants == 0 or self.n_haps == 0

    @property
    def all(self) -> sparse.csc_matrix:
        """Get unsliced sparse matrix of all boolean genotypes"""
        return sparse.vstack(
            [self._load_haplotypes(chunk) for chunk in self.chunks[:, 0]]
        )

    def range(
        self, min_bp: int, max_bp: int, inclusive: bool = True
    ) -> sparse.csc_matrix:
        """Get sparse matrix of boolean genotypes in position range"""
        if inclusive:
            max_bp += 1
        positions = np.array([variant[1] for variant in self.variants], dtype=int)
        return self[positions.searchsorted(min_bp) : positions.searchsorted(max_bp), :]
