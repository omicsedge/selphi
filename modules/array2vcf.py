from typing import List, Union
import subprocess
from pathlib import Path
from uuid import uuid4
from shutil import copyfileobj

import numpy as np
from scipy import sparse


class VcfWriter:
    """Class to convert from numpy array to VCF"""

    def __init__(
        self, target_sample_names: List[str], chromomsome: str, tmpdir: Union[Path, str]
    ):
        self.target_sample_names = target_sample_names
        self.chromosome = chromomsome
        self.tmpdir = Path(tmpdir).joinpath(str(uuid4()))
        self.tmpdir.mkdir(parents=True, exist_ok=True)

    @property
    def num_samples(self) -> int:
        """Get number of samples"""
        return len(self.target_sample_names)

    @property
    def num_haplotypes(self) -> int:
        """Get number of samples"""
        return self.num_samples * 2

    @property
    def chr_length(self) -> str:
        """Get chromosome length for VCF"""
        length_chrs_hg38 = {
            "1": "248956422",
            "2": "242193529",
            "3": "198295559",
            "4": "190214555",
            "5": "181538259",
            "6": "170805979",
            "7": "159345973",
            "8": "145138636",
            "9": "138394717",
            "10": "133797422",
            "11": "135086622",
            "12": "133275309",
            "13": "114364328",
            "14": "107043718",
            "15": "101991189",
            "16": "90338345",
            "17": "83257441",
            "18": "80373285",
            "19": "58617616",
            "20": "64444167",
            "21": "46709983",
            "22": "50818468",
            "X": "156040895",
            "Y": "57227415",
            "MT": "16569",
        }
        return length_chrs_hg38.get(self.chromosome, "")

    @staticmethod
    def compress(file: Path):
        subprocess.run(["bgzip", "-f", file], check=True)

    @staticmethod
    def index_vcf(vcf_file: Union[Path, str]):
        subprocess.run(["tabix", "-f", "-p", "vcf", vcf_file], check=True)

    def write_header(self) -> Path:
        header = (
            "##fileformat=VCFv4.2\n"
            "##source=SELPHI_v1.0_1June2023.py Selfdecode™\n"
            '##FILTER=<ID=PASS,Description="All filters passed">\n'
            '##INFO=<ID=IMP,Number=0,Type=Flag,Description="Imputed marker">\n'
            '##INFO=<ID=AF,Number=A,Type=Float,Description="Estimated ALT Allele Frequencies">\n'
            '##INFO=<ID=AN,Number=1,Type=Integer,Description="Allele Number">\n'
            '##INFO=<ID=AC,Number=1,Type=Integer,Description="Estimated Allele Count">\n'
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            '##FORMAT=<ID=DS,Number=A,Type=Float,Description="estimated ALT dose">\n'
            '##FORMAT=<ID=AP1,Number=A,Type=Float,Description="estimated ALT dose on first haplotype">\n'
            '##FORMAT=<ID=AP2,Number=A,Type=Float,Description="estimated ALT dose on second haplotype">\n'
            f"##contig=<ID={self.chromosome},length={self.chr_length},assembly=GRCh38,species=HomoSapiens>\n"
        )
        columns = (
            f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(self.target_sample_names)
            + "\n"
        )
        self.tmpdir.joinpath("header.vcf").write_text(header + columns)
        self.compress(self.tmpdir.joinpath("header.vcf"))
        return self.tmpdir.joinpath("header.vcf.gz")

    def write_variants(
        self,
        all_variant_ids: Union[List, np.ndarray],
        target_ids: Union[List, np.ndarray],
        sparse_probs: sparse.csc_matrix,
    ) -> Path:
        out_path = self.tmpdir.joinpath(str(uuid4()))
        target_ids = set(target_ids)

        AN = self.num_haplotypes

        with out_path.open("w") as fout:
            # Write to file in chunks of 1000 variants
            for start in range(0, sparse_probs.shape[0], 1000):
                probs = sparse_probs[start : start + 1000, :].toarray()
                variant_ids = all_variant_ids[start : start + 1000]

                # load metrics
                AP1 = probs[:, 0::2]
                AP2 = probs[:, 1::2]
                DS = AP1 + AP2
                GT1 = (AP1 > 0.5).astype(int)
                GT2 = (AP2 > 0.5).astype(int)
                AC = np.sum(GT1 + GT2, axis=1)
                AF = [
                    int(ele) if ele % 1 == 0 else np.round(ele, 4) for ele in (AC / AN)
                ]

                # Write variant records
                for i, idx in enumerate(variant_ids):
                    chrom, pos, ref, alt = idx.split("-")
                    vcf_INFO = f"AF={AF[i]};AC={AC[i]};AN={AN}"
                    if i not in target_ids:
                        vcf_INFO += ";IMP"
                    fout.write(
                        "\t".join(
                            [
                                chrom,
                                pos,
                                idx,
                                ref,
                                alt,
                                ".\tPASS",
                                vcf_INFO,
                                "GT:DS:AP1:AP2\t",
                            ]
                        )
                    )
                    # Convert float numbers to integers if they are integers
                    DS_int = [
                        "{:.2f}".format(num).rstrip("0").rstrip(".")
                        if num % 1 != 0
                        else int(num)
                        for num in DS[i, :]
                    ]
                    AP1_int = [
                        "{:.2f}".format(num).rstrip("0").rstrip(".")
                        if num % 1 != 0
                        else int(num)
                        for num in AP1[i, :]
                    ]
                    AP2_int = [
                        "{:.2f}".format(num).rstrip("0").rstrip(".")
                        if num % 1 != 0
                        else int(num)
                        for num in AP2[i, :]
                    ]
                    # Create the formatted string for vcf_GT
                    fout.write(
                        "\t".join(
                            [
                                f"{gt1}|{gt2}:{DS_s}:{AP1_s}:{AP2_s}"
                                for gt1, gt2, DS_s, AP1_s, AP2_s in zip(
                                    GT1[i, :], GT2[i, :], DS_int, AP1_int, AP2_int
                                )
                            ]
                        )
                    )
                    fout.write("\n")

        self.compress(out_path)
        return out_path.with_suffix(".gz")

    def complete_vcf(self, filelist: List[Path], vcf_file: Union[Path, str]) -> Path:
        if str(vcf_file).endswith(".vcf.gz"):
            out_path = Path(vcf_file)
        else:
            out_path = Path(vcf_file).with_suffix(".vcf.gz")
        with out_path.open("wb") as fout:
            with self.tmpdir.joinpath("header.vcf.gz").open("rb") as header:
                copyfileobj(header, fout)
            for file in filelist:
                with file.open("rb") as fin:
                    copyfileobj(fin, fout)
        self.index_vcf(out_path)
        return out_path
