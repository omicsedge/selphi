"""
Sparse Matrix VCF Converter

The Sparse Matrix VCF Converter is a Python class that allows
to convert a sparse matrix containing genotype information to the Variant Call Format (VCF) file format. 

Usage:
1. Create an instance of the SparseMatrixVCFConverter class by passing a sparse matrix as input:
    converter = SparseMatrixVCFConverter(sparse_matrix)

2. Call the convert_to_vcf method to generate the VCF file from the sparse matrix:
    converter.convert_to_vcf(output_file)
    - output_file: The path to the output VCF file that will be generated.
"""

import subprocess
import numpy as np
from tqdm import tqdm


class Sparse2vcf:
    """Class to convert from sparse matrix to VCF"""

    def __init__(self, results, target_sample_names, reference_ids, target_ids):
        self.results = results > 0.5
        self.target_sample_names = target_sample_names
        self.reference_ids = reference_ids
        self.target_ids = target_ids  # ndarray of indexes in target
        self.probs = results

    @property
    def chromosome(self) -> str:
        """Get chromosome"""
        first_target_id = self.reference_ids[0]
        chrom = first_target_id.split("-")[0]
        return chrom

    @property
    def num_haploids(self) -> int:
        """Get number of samples"""
        return self.results.shape[0]

    @property
    def num_samples(self) -> int:
        """Get number of samples"""
        return int(self.results.shape[0] / 2)

    @property
    def num_variants(self) -> int:
        """Get number of variants"""
        return self.results.shape[1]

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
        return length_chrs_hg38[self.chromosome]

    @property
    def convert_sparse_to_matrix(self) -> np.array:
        """Convert sparse matrix to dense array"""
        return self.results.toarray().astype(np.uint8)

    @property
    def get_gt_paternal(self) -> np.array:
        """Get genotype calls for paternal haplotype"""
        return self.convert_sparse_to_matrix[0::2, :]

    @property
    def get_gt_maternal(self) -> np.array:
        """Get genotype calls for paternal haplotype"""
        return self.convert_sparse_to_matrix[1::2, :]

    @property
    def allele_sum(self) -> np.array:
        """Get allele sum paternal and maternal"""
        return self.get_gt_paternal + self.get_gt_maternal

    @property
    def get_genotype_format_VCF(self) -> np.array:
        """Get genotyepe calls for VCF output"""
        return np.where(
            self.allele_sum == 0,
            "0|0",
            np.where(
                self.allele_sum == 2,
                "1|1",
                np.where(self.get_gt_paternal == 1, "1|0", "0|1"),
            ),
        )

    @property
    def get_AN(self) -> int:
        """Get AN allele number"""
        return self.num_haploids

    @property
    def get_AC(self) -> np.array:
        """Get AC allele count"""
        return np.sum(self.allele_sum, axis=0)

    @property
    def get_AF(self) -> np.array:
        """Get ALT allele freq"""
        Afreq = self.get_AC / self.get_AN
        return [int(ele) if ele % 1 == 0 else np.round(ele, 4) for ele in Afreq]

    @property
    def get_RA(self) -> np.array:
        """Get RA prob"""
        return

    @property
    def get_RR(self) -> np.array:
        """Get RR prob"""
        return

    @property
    def get_AA(self) -> np.array:
        """Get AA prob"""
        return

    @property
    def get_DR2(self) -> np.array:
        """Get DR2 imputation score"""
        return

    @property
    def convert_probs_to_matrix(self) -> np.array:
        """Convert sparse matrix to dense array"""
        return self.probs.toarray()

    @property
    def get_AP1(self) -> np.array:
        """Get AP ALT dose on first haplotype"""
        return self.convert_probs_to_matrix[0::2, :]

    @property
    def get_AP2(self) -> np.array:
        return self.convert_probs_to_matrix[1::2, :]

    @property
    def get_DS(self) -> np.array:
        """Get DS ALT dose"""
        return self.get_AP1 + self.get_AP2

    def convert_to_vcf(self, output_file):
        with open(output_file, "w", buffering=1048576) as vcf_file:
            # Write VCF header
            vcf_file.write(
                "##fileformat=VCFv4.2\n"
                "##source=SELPHI_v1.0_1June2023.py Selfdecode™\n"
                '##FILTER=<ID=PASS,Description="All filters passed">\n'
                '##INFO=<ID=IMP,Number=0,Type=Flag,Description="Imputed marker">\n'
                '##INFO=<ID=AF,Number=A,Type=Float,Description="Estimated ALT Allele Frequencies">\n'
                '##INFO=<ID=AN,Number=A,Type=Float,Description="Allele Number">\n'
                '##INFO=<ID=AC,Number=A,Type=Float,Description="Estimated Allele Count">\n'
                # '##INFO=<ID=DR2,Number=A,Type=Float,Description="Dosage R-Squared: estimated squared correlation between estimated REF dose [P(RA) + 2*P(RR)]" and the true REF dose>\n'
                '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                '##FORMAT=<ID=DS,Number=A,Type=Float,Description="estimated ALT dose">\n'
                '##FORMAT=<ID=AP1,Number=A,Type=Float,Description="estimated ALT dose on first haplotype">\n'
                '##FORMAT=<ID=AP2,Number=A,Type=Float,Description="estimated ALT dose on second haplotype">\n'
                f"##contig=<ID={self.chromosome},length={self.chr_length},assembly=GRCh38,species=HomoSapiens>\n"
            )
            # Write sample IDs
            vcf_file.write(
                f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(self.target_sample_names)
                + "\n"
            )

            # load metrics
            # DR2 = self.get_DR2
            AF = self.get_AF
            AC = self.get_AC
            GT = self.get_genotype_format_VCF
            DS = self.get_DS
            AP1 = self.get_AP1
            AP2 = self.get_AP2

            # Write variant records
            for i, idx in enumerate(
                tqdm(
                    self.reference_ids,
                    desc=(
                        f" [output] Writing {self.num_samples} sample(s) for "
                        f"{self.num_variants} variants"
                    ),
                    ncols=75,
                    bar_format="{desc}:\t\t\t{percentage:3.0f}% in {elapsed}",
                    colour="#808080",
                )
            ):
                chrom, pos, ref, alt = idx.split("-")
                # Convert float numbers to integers if they are integers
                DS_int = [
                    int(num) if num % 1 == 0 else np.round(num, 3) for num in DS[:, i]
                ]
                AP1_int = [
                    int(num) if num % 1 == 0 else np.round(num, 3) for num in AP1[:, i]
                ]
                AP2_int = [
                    int(num) if num % 1 == 0 else np.round(num, 3) for num in AP2[:, i]
                ]
                # Create the formatted string for vcf_GT
                vcf_FORMAT = "\t".join(
                    [
                        f"{GT_s}:{DS_s}:{AP1_s}:{AP2_s}"
                        for GT_s, DS_s, AP1_s, AP2_s in zip(
                            GT[:, i], DS_int, AP1_int, AP2_int
                        )
                    ]
                )
                vcf_INFO = f"AF={AF[i]};AC={AC[i]}"
                if i not in self.target_ids:
                    vcf_INFO += ";IMP"
                vcf_file.write(
                    f"{chrom}\t{pos}\t{idx}\t{ref}\t{alt}\t.\tPASS\t{vcf_INFO}\t"
                    f"GT:DS:AP1:AP2\t{vcf_FORMAT}\n"
                )

    def compress_vcf(self, vcf_file):
        subprocess.run(["bgzip", "-f", vcf_file], check=True)

    def index_vcf(self, vcf_file):
        subprocess.run(["tabix", "-f", "-p", "vcf", vcf_file], check=True)
