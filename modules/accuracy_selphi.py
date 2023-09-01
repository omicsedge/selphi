import os
import uuid
import subprocess
from io import BytesIO
from typing import List, Tuple, Union, Dict
from scipy import sparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from shutil import copyfileobj,rmtree
import concurrent.futures
import warnings
import natsort as nats

class DataIngestion:

    def __init__(self, filepath: str, chunk_size: int = 10_000, samples_file: str = '') -> None:
        """Initialize the SparseComparison class"""
        self.filepath: str = self.initial_check(filepath)
        self.chunk_size: int = chunk_size
        self.chunk_ranges: List[Tuple[int, int]] = self.get_ranges(self.filepath)
        self.samples_file: str = samples_file

    def initial_check(self, vcf_path: str) -> str:
        """Checks and indexes the provided VCF file if needed."""
        if not os.path.exists(vcf_path):
            raise FileNotFoundError(f"Missing input file: {vcf_path}")
        if not (os.path.exists(vcf_path + ".tbi") or os.path.exists(vcf_path + ".csi")):
            print(f"Indexing input file: {vcf_path}")
            subprocess.run(f"tabix {vcf_path}", shell=True)
        return vcf_path

    def get_ranges(self, vcf_path: str) -> List[Tuple[int, int]]:
        """Get the range of positions from the VCF file."""
        cmd = [
            'bcftools', 'index', '--stats', vcf_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise ValueError(f"Error executing bcftools: {result.stderr}")
        lines = [line.split('\t') for line in result.stdout.splitlines()]
        if len(lines) > 1:
            raise ValueError("Only one chromosome per file is supported")
        chrom, chr_length, num_variants = lines[0]
        if chr_length == '.':
            chr_length = self.get_chromosome_length(chrom)
        self.chrom = chrom
        return self.ranges(chrom, int(chr_length), int(num_variants))

    def ranges(self, chrom: str, chr_length: int, num_variants: int) -> List[Tuple[int, int]]:
        bp_per_variant = chr_length / num_variants
        bp_per_chunk = bp_per_variant * self.chunk_size
        current_start = 1
        ranges = []
        while current_start < chr_length:
            end = min(current_start + bp_per_chunk, chr_length)
            ranges.append((int(current_start), int(end)))
            current_start = end + 1
        ranges[-1] = (ranges[-1][0], 999_999_999)
        return ranges

    def stdout_to_sparse(self, result):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return sparse.csc_matrix(
                np.loadtxt(BytesIO(result.stdout), delimiter="|", dtype=np.bool_)
            )

    def _ingest_data(self, vcf_path: str, chrom: str, start: int, end: int):
        """Load data vars and haplotypes from the VCF file for the given range"""
        
        args = (start, end, chrom, vcf_path)
        
        def _load_haplotypes(args):
            start, end, chrom, vcf_path = args
            if self.samples_file != '':
                commands = [(
                    f"bcftools view -r {chrom}:{start}-{end} "
                    f"{vcf_path} -S {self.samples_file} | bcftools query -f '[|%GT]\n' "
                    f"| sed s'/|//' | sed s'/\//|/'g"
                )]
            else:
                commands = [(
                    f"bcftools view -r {chrom}:{start}-{end} "
                    f"{vcf_path} | bcftools query -f '[|%GT]\n' "
                    f"| sed s'/|//' | sed s'/\//|/'g"
                )]
            result = subprocess.run(commands, shell=True, stdout=subprocess.PIPE)
            return self.stdout_to_sparse(result)

        def _load_variants(args):
            start, end, chrom, vcf_path = args
            cmd = [
                'bcftools', 'query', '-r', f'{chrom}:{start}-{end}', 
                '-f', '%CHROM-%POS-%REF-%ALT\n', vcf_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            return [line.strip() for line in result.stdout.splitlines()]

        def _load_samples(args):
            
            vcf_path = args[3]
            if self.samples_file != '':
                cmd = [
                    'cat', f'{self.samples_file}',
                ]
            else:
                cmd = [
                    'bcftools', 'query', '-l', vcf_path
                ]                
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            return [line.strip() for line in result.stdout.splitlines()]

        def _load_(args):
            return _load_variants(args), _load_haplotypes(args), _load_samples(args)

        return _load_(args)

    def get_chromosome_length(self, chrom) -> str:
        """Get chromosome length for VCF"""
        length_chrs_hg38 = {
            "1": "248956422", "2": "242193529", "3": "198295559",
            "4": "190214555", "5": "181538259", "6": "170805979",
            "7": "159345973", "8": "145138636", "9": "138394717",
            "10": "133797422", "11": "135086622", "12": "133275309",
            "13": "114364328", "14": "107043718", "15": "101991189",
            "16": "90338345", "17": "83257441", "18": "80373285",
            "19": "58617616", "20": "64444167", "21": "46709983",
            "22": "50818468", "X": "156040895", "Y": "57227415",
            "MT": "16569",
        }
        return length_chrs_hg38.get(chrom, "")

class CalculateMetrics:

    def __init__(
        self, imputed: sparse.csc_matrix, wgs: sparse.csc_matrix, 
        vID_imputed: List[str], vID_wgs: List[str],
        sID_imputed: List[str], sID_wgs: List[str], 
        chunk_index: int, unique_folder_name: str) -> None:

        self.imputed = imputed.toarray().astype(int)
        self.wgs = wgs.toarray().astype(int)
        self.vID_imputed = vID_imputed
        self.vID_wgs = vID_wgs
        self.sID_imputed = sID_imputed
        self.sID_wgs = sID_wgs
        self.sample_size = self.imputed.shape[1]
        self.variant_size = self.imputed.shape[0]
        self.unique_folder_name = unique_folder_name
        self.chunk_index = chunk_index
        self.dose_to_binary = np.array([[1, 0],[1, 1],[0, 1]])
        if not os.path.exists(self.unique_folder_name):
            os.makedirs(self.unique_folder_name, exist_ok=True)


    def _load_dosage(self, data):
        return data[:, ::2] + data[:, 1::2]

    def _compute_metrics(self, axis: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        imputed = self.dose_to_binary[self._load_dosage(self.imputed).flatten()].reshape(-1, self.sample_size)
        wgs = self.dose_to_binary[self._load_dosage(self.wgs).flatten()].reshape(-1, self.sample_size)

        TP = np.sum(np.logical_and(imputed == 1, wgs == 1), axis=axis)
        TN = np.sum(np.logical_and(imputed == 0, wgs == 0), axis=axis)
        FP = np.sum(np.logical_and(imputed == 1, wgs == 0), axis=axis)
        FN = np.sum(np.logical_and(imputed == 0, wgs == 1), axis=axis)

        return TP, TN, FP, FN

    def save_fscore(self):
        
        filename = f'fscore_res_to_join_{self.chunk_index}.csv'
        fn, ex = os.path.splitext(filename)
        fn_per_sample = os.path.join(self.unique_folder_name, f"{fn}_x_sample{ex}")
        fn_per_variant = os.path.join(self.unique_folder_name, f"{fn}_x_variant{ex}")

        fscore = self.fscore()

        df_var = pd.DataFrame({
            'snpID' : self.vID_imputed,
            'TP': fscore.get('var_TP', []),
            'TN': fscore.get('var_TN', []),
            'FP': fscore.get('var_FP', []),
            'FN': fscore.get('var_FN', []),
            'TE': fscore.get('var_TE', []),
            'F-score': fscore.get('f-score', []),
        })

        df_var.applymap(self._format_number).to_csv(fn_per_variant, index=False)

        df_sample = pd.DataFrame({
            'SampleID': self.sID_imputed,
            'TP': fscore.get('TP', []),
            'TN': fscore.get('TN', []),
            'FP': fscore.get('FP', []),
            'FN': fscore.get('FN', []),
            'TE': fscore.get('TE', []),
        })

        df_sample.applymap(self._format_number).to_csv(fn_per_sample, index=False)

        del df_var
        del df_sample

    def fscore(self) -> Dict[str, Union[List[float], np.ndarray]]:
        """return F-score"""
        results = {}

        for axis in [0, 1]:
            TP, TN, FP, FN = self._compute_metrics(axis)
            
            if axis == 1:
                precision = np.divide(TP + 1e-8, np.add(TP, FP) + 1e-8)
                recall = np.divide(TP + 1e-8, np.add(TP, FN) + 1e-8)
                f1 = np.round( 2 * np.divide(np.multiply(precision, recall), 
                                np.add(precision, recall)
                                ),decimals=6)

                results['f-score'] = f1
                results.update({
                    'var_TP': TP,
                    'var_FP': FP,
                    'var_FN': FN,
                    'var_TN': TN,
                    'var_TE': FP+FN,
                    'var_precision': np.round(precision, decimals=6),
                    'var_recall': np.round(recall, decimals=6)
                })

            else:
                reshaped_length = int(len(TP) / 2)
                metrics = [TP, FP, FN, TN]
                metrics = [np.sum(m.reshape(reshaped_length,2),axis=1) for m in metrics]

                results.update({
                    'TP': metrics[0],
                    'FP': metrics[1],
                    'FN': metrics[2],
                    'TN': metrics[3],
                    'TE': metrics[1]+metrics[2],
                })

        return results

    def save_accuracy(self):
        
        filename = f'accuracy_res_to_join_{self.chunk_index}.csv'
        fn, ex = os.path.splitext(filename)
        fn_per_sample = os.path.join(self.unique_folder_name, f"{fn}_x_sample{ex}")
        fn_per_variant = os.path.join(self.unique_folder_name, f"{fn}_x_variant{ex}")

        accuracy = self.accuracy()

        df_var = pd.DataFrame({
            'Accuracy': accuracy.get('accuracy_per_var', []),
        })

        df_var.applymap(self._format_number).to_csv(fn_per_variant, index=False)

        df_sample = pd.DataFrame({
            'Accuracy': accuracy.get('correct_pred_per_sample', []),
            'N_var': accuracy.get('N_var', []),
        })

        df_sample.applymap(self._format_number).to_csv(fn_per_sample, index=False)

        del df_var
        del df_sample

    def accuracy(self) -> Dict[str, np.ndarray]:
        """return Accuracy/Concordance_P0"""
        correct_prediction = np.equal(self.imputed, self.wgs)
        accuracy_per_var = np.mean(correct_prediction.astype(float), axis=1)
        correct_pred_sum = np.sum(correct_prediction.astype(float), axis=0)
        correct_pred_per_sample = np.divide(correct_pred_sum.reshape(-1, 2).sum(axis=1), 2.0)
        
        del correct_pred_sum
        del correct_prediction

        return {
            'accuracy_per_var': np.round(accuracy_per_var, decimals=6),
            'correct_pred_per_sample': correct_pred_per_sample,
            'N_var': self.variant_size,
        }

    def compute_IQS(self) -> dict:
        # dosage to probability mapping
        dose_to_probability = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1]}
        
        # Convert dosages to probabilities
        x = np.array([dose_to_probability[d] for d in imputed.ravel()]).reshape(imputed.shape[0], -1)
        y = np.array([dose_to_probability[d] for d in wgs.ravel()]).reshape(wgs.shape[0], -1)
        
        # Calculate P0 (observed concordance)
        correct_prediction = np.equal(x, y)
        P0 = np.mean(correct_prediction.astype(float), axis=1)
        
        # Compute expected concordance Pc using formula based on the definitions
        Ns = len(y[0]) / 3
        p11 = np.multiply(y[:, 0::3], x[:, 0::3]).sum(axis=1)
        p22 = np.multiply(y[:, 1::3], x[:, 1::3]).sum(axis=1)
        p33 = np.multiply(y[:, 2::3], x[:, 2::3]).sum(axis=1)
        N1_i = p11 + p22 + p33
        Pc = (N1_i / Ns) ** 2
        
        # Calculate IQS
        IQS = (P0 - Pc) / (1 - Pc)
        
        return {
            'P0_per_var': P0,
            'IQS': IQS
        }

    def save_r2(self):
        
        filename = f'r2_res_to_join_{self.chunk_index}.csv'
        fn, ex = os.path.splitext(filename)
        fn_per_variant = os.path.join(self.unique_folder_name, f"{fn}_x_variant{ex}")
        fn_per_sample = os.path.join(self.unique_folder_name, f"{fn}_x_sample{ex}")

        r2 = self.r2()

        df_var = pd.DataFrame({
            'r2': r2.get('r2_per_variant', []),
        })

        df_var.applymap(self._format_number).to_csv(fn_per_variant, index=False)

        df_sample = pd.DataFrame({
            'r2_per_sample_num': r2.get('r2_per_sample_num', []),
            'r2_per_sample_den': r2.get('r2_per_sample_den', []),
        })

        df_sample.applymap(self._format_number).to_csv(fn_per_sample, index=False)

        del df_var

    def r2(self) -> Dict[str, np.ndarray]:
        """return Pearson r2 for both per-variant and per-sample"""
        
        imputed = self._load_dosage(self.imputed)
        wgs = self._load_dosage(self.wgs)

        N = len(imputed[0])
        
        # Per-variant calculations
        x_sum_var = np.sum(imputed, axis=1)
        y_sum_var = np.sum(wgs, axis=1)
        xy_sum_var = np.sum(imputed * wgs, axis=1)
        x_squared_sum_var = np.sum(imputed ** 2, axis=1)
        y_squared_sum_var = np.sum(wgs ** 2, axis=1)

        num_var = N * xy_sum_var - x_sum_var * y_sum_var
        den_var = np.sqrt((N * x_squared_sum_var - x_sum_var ** 2) * (N * y_squared_sum_var - y_sum_var ** 2)) + 1e-15
        r2_per_variant = np.divide(num_var, den_var) ** 2
        r2_per_variant = np.round(r2_per_variant, decimals=6)

        # Per-sample calculations
        x_sum_sample = np.sum(imputed, axis=0)
        y_sum_sample = np.sum(wgs, axis=0)
        xy_sum_sample = np.sum(imputed * wgs, axis=0)
        x_squared_sum_sample = np.sum(imputed ** 2, axis=0)
        y_squared_sum_sample = np.sum(wgs ** 2, axis=0)

        num_sample = (len(imputed) * xy_sum_sample - x_sum_sample * y_sum_sample) + 1e-15
        den_sample = np.sqrt((len(imputed) * x_squared_sum_sample - x_sum_sample ** 2) * 
                             (len(imputed) * y_squared_sum_sample - y_sum_sample ** 2)) + 1e-15
        
        # r2_per_sample = np.divide(num_sample, den_sample) ** 2
        # r2_per_sample = np.round(r2_per_sample, decimals=6)

        return {
            'r2_per_variant': r2_per_variant,
            'r2_per_sample_num': num_sample,
            'r2_per_sample_den': den_sample,
        }

    def save_rmse(self):
        
        filename = f'RMSE_res_to_join_{self.chunk_index}.csv'
        fn, ex = os.path.splitext(filename)
        fn_per_variant = os.path.join(self.unique_folder_name, f"{fn}_x_variant{ex}")

        rmse = self.rmse()

        df_var = pd.DataFrame({
            'RMSE': rmse.get('var_rmse', []),
        })

        df_var.applymap(self._format_number).to_csv(fn_per_variant, index=False)

        del df_var


    def rmse(self) -> Dict[str, np.ndarray]:
        """return RMSE for both per-variant and per-sample"""

        # squared error
        se = (self.imputed - self.wgs) ** 2

        # root mean squared error per variant
        var_rmse = np.sqrt(np.mean(se, axis=1))
        
        # squared error summed per sample
        # sample_rmse = np.sqrt(np.mean(se, axis=0))
        return {
            'var_rmse': var_rmse,
            # 'sample_rmse': sample_rmse
        }

    def _format_number(self, x):
        if isinstance(x, int):
            return str(x)
        elif isinstance(x, float):
            return str(int(x)) if x.is_integer() else str(x)
        return x

    def custom_sort(self, filenames):
        keywords = ['fscore_res', 'accuracy_res', 'r2_res_to', 'rmse_res']
        mapping = {keyword: '' for keyword in keywords}

        for path in filenames:
            for keyword in keywords:
                if keyword in path:
                    mapping[keyword] = path
                    break

        return [mapping[keyword] for keyword in keywords]

    def _get_sorted_csv_files(self, suffix: str):
        """Get the sorted csv files based on the given suffix."""
        csv_files = [os.path.join(self.unique_folder_name, file)
                     for file in os.listdir(self.unique_folder_name) if file.endswith(suffix)]
        return self.custom_sort(csv_files)

    def concatenate_files(self) -> str:
        fn_per_sample = os.path.join(self.unique_folder_name, f'saved_chunked_tmp_{self.chunk_index}_x_sample.csv')
        fn_per_variant = os.path.join(self.unique_folder_name, f'saved_chunked_tmp_{self.chunk_index}_x_variant.csv')

        # Concatenate variant files
        csv_files_variant = self._get_sorted_csv_files(f'_res_to_join_{self.chunk_index}_x_variant.csv')
        cmd_variant = f"paste -d ',' {' '.join(csv_files_variant)} > {fn_per_variant}"
        subprocess.run(cmd_variant, shell=True)

        # Concatenate sample files
        csv_files_sample = self._get_sorted_csv_files(f'_res_to_join_{self.chunk_index}_x_sample.csv')
        cmd_sample = f"paste -d ',' {' '.join(csv_files_sample)} > {fn_per_sample}"
        subprocess.run(cmd_sample, shell=True)

        return fn_per_sample, fn_per_variant

    def save_metrics(self):
        self.save_fscore(), self.save_accuracy(), self.save_r2(), self.save_rmse()
        return self.concatenate_files()

def remove_duplicates(input_file, unique_folder_name):
    temp_file = os.path.join(unique_folder_name, "temp_file_duplicates.txt")
    cmd = f"awk '!seen[$0]++' {input_file} > {temp_file} && mv {temp_file} {input_file}"
    subprocess.run(cmd, shell=True, check=True)

def concatenate_samples(file_paths: List[str]) -> pd.DataFrame:
    sample_data = []

    def read_sample_data(sample_file):
        return pd.read_csv(sample_file)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for sample_data_chunk in executor.map(read_sample_data, [sample_file for sample_file in file_paths]):
            sample_data.append(sample_data_chunk)

    concatenated_sample_df = pd.concat(sample_data, ignore_index=True)
    aggregate_df = concatenated_sample_df.groupby("SampleID").agg({
        "Accuracy": "sum",
        "N_var": "sum",
        "r2_per_sample_num": "sum",
        "r2_per_sample_den": "sum",
        "TP": "sum",
        "TN": "sum",
        "FP": "sum",
        "FN": "sum",
        "TE": "sum"
    }).reset_index()

    aggregate_df["Accuracy"] = np.round(aggregate_df["Accuracy"] / aggregate_df["N_var"],decimals=6)
    aggregate_df['r2'] = np.round(np.divide(aggregate_df['r2_per_sample_num'], aggregate_df['r2_per_sample_den']) ** 2, decimals=6)
    aggregate_df["Precision"] = np.round(aggregate_df["TP"] / (aggregate_df["TP"] + aggregate_df["FP"]),decimals=6)
    aggregate_df["Recall"] = np.round(aggregate_df["TP"] / (aggregate_df["TP"] + aggregate_df["FN"]),decimals=6)
    aggregate_df["F1-score"] = np.round(2 * aggregate_df["Precision"] * aggregate_df["Recall"] / (aggregate_df["Precision"] + aggregate_df["Recall"]),decimals=6)

    def _format_number(x):
        if isinstance(x, int):
            return str(x)
        elif isinstance(x, float):
            return str(int(x)) if x.is_integer() else str(x)
        return x

    aggregate_df.drop(['N_var','r2_per_sample_num','r2_per_sample_den'],axis=1,inplace=True)

    aggregate_df = aggregate_df[["SampleID","Accuracy","r2","TP","TN","FP","FN","TE","Precision","Recall","F1-score"]]

    return aggregate_df.applymap(_format_number)

def calculate_metrics(chunk_index, imputed, wgs, chrom, start, end, unique_folder_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vID_imputed, Haps_imputed, sID_imputed = imputed._ingest_data(imputed.filepath, chrom, start, end)
        vID_wgs, Haps_wgs, sID_wgs = wgs._ingest_data(wgs.filepath, chrom, start, end)
        if len(vID_imputed) == 0 and len(vID_wgs) == 0:
            return None,None
        else:
            return CalculateMetrics(Haps_imputed, Haps_wgs, vID_imputed, vID_wgs, sID_imputed, sID_wgs, chunk_index, unique_folder_name).save_metrics()

def main():
    parser = argparse.ArgumentParser(usage='%(prog)s --imputed <imputed_file> --wgs <whole_genome_file>\nUse -h or --help to display help.')
    parser.add_argument('--wgs', dest='wgs', type=str, required=True, help='path to whole genome file')
    parser.add_argument('--imputed', dest='imputed', type=str, required=True, help='path to imputed file')
    parser.add_argument('-S', dest='samples_file', type=str, required=False, default='', help='File of sample names to include in the analysis. One sample per line.')
    parser.add_argument('-c', dest='chunk_size', type=int, required=False, default=10000, help='size of data chunks, [ default: 10000 ]')
    parser.add_argument('--threads', dest='threads', type=int, required=False, default=10, help='threads to use for parallel processing, [ default: 10 ]')
    parser.add_argument('--keep_tmp', dest='keep_tmp', action='store_true', help='If set, temporary files will not be deleted.')

    args = parser.parse_args()

    imputedpath = args.imputed
    wgspath = args.wgs
    threads = args.threads
    chunk_size = args.chunk_size
    samples_file = args.samples_file

    imputed = DataIngestion(imputedpath, chunk_size, samples_file)
    wgs = DataIngestion(wgspath, chunk_size, samples_file)

    chunk_ranges = imputed.chunk_ranges

    # Extract the base name without the extension and directory path
    base_name = os.path.basename(imputedpath).split('.')[0]
    directory_path = os.path.dirname(imputedpath) or "."
    variant_out_path = f'{directory_path}/{base_name}_accuracy_variants.csv'
    sample_out_path = f'{directory_path}/{base_name}_accuracy_samples.tsv'

    unique_folder_name = '/tmp/tmp_accuracy/' + str(uuid.uuid4())
    print(f'tmp files: {unique_folder_name}')
    print(f'out files: {os.path.abspath(directory_path)}')

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(calculate_metrics, chunk[0], imputed, wgs, imputed.chrom, chunk[1][0], chunk[1][1], unique_folder_name) for chunk in enumerate(chunk_ranges)]
        for future in tqdm(concurrent.futures.as_completed(futures), 
                           total=len(futures), 
                           desc="🍺 \033[32mCalculating accuracy in chunks:\033[0m",
                           ncols=75,
                           bar_format="{desc}:\t\t{percentage:3.0f}% in {elapsed}",
            ):
            pass

    file_paths = [result.result() for result in futures]

    file_paths = list(filter(lambda x: None not in x, file_paths))

    variant_file_paths = nats.natsorted([variant_file for _, variant_file in file_paths])
    sample_file_paths = nats.natsorted([sample_file for sample_file, _ in file_paths])

    with open(variant_out_path, "wb") as variant_fout:
        first_file = True
        for variant_file in variant_file_paths:
            with open(variant_file, "rb") as variant_fin:
                if first_file:
                    # Copy the entire file if it's the first file
                    copyfileobj(variant_fin, variant_fout)
                    first_file = False
                else:
                    # Skip the header line and copy the rest of the file for subsequent files
                    variant_fin.readline()
                    copyfileobj(variant_fin, variant_fout)

    #drop potential duplicates between chunks
    remove_duplicates(variant_out_path,unique_folder_name)
    #pd.read_csv(variant_out_path).drop_duplicates().to_csv(variant_out_path,index=False)
    concatenated_sample_df = concatenate_samples(sample_file_paths)
    concatenated_sample_df.to_csv(sample_out_path, index=False, sep='\t')

    if not args.keep_tmp:
        rmtree(os.path.join(unique_folder_name))

if __name__ == '__main__':
    main()
