from typing import List, Tuple, Union, TypeVar
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt



CHECKPOINTS_FILENAME = 'checkpoints.tsv'
MEMUSAGE_FILENAME = 'memusage.tsv'


def B_to_MB(bytes: Union[int, float, np.float64, np.ndarray]):
    return bytes / 1_000_000

def ms_to_min(milliseconds: Union[int, float, np.float64, np.ndarray]):
    return milliseconds / 60_000

floats = TypeVar('floats', float, np.float64, np.ndarray)
def B_ms_to_GB_min(bytes_miliseconds: floats) -> floats:
    return bytes_miliseconds / 1_000_000_000 / 60_000 # type: ignore



class benchmark_logs():

    mark_timestamps: List[np.float64]
    mark_comment: List[str]
    mark_marker: List[str]

    name: str
    dirname: str
    color: Union[str, None]

    memusage: np.ndarray
    exec_time: np.float64
    peakmemusage: np.float64
    exec_start_timestamp: np.float64

    interpolation_sections_start: List[np.float64] = []
    interpolation_sections_end: List[np.float64] = []


    def __init__(self, name: str, dirname: str, color: Union[str, None] = None):
        self.mark_timestamps: List[np.float64] = []
        self.mark_comment: List[str] = []
        self.mark_marker: List[str] = []

        self.name = name
        self.dirname = dirname
        self.color = color

        self.interpolation_sections_start = []
        self.interpolation_sections_end = []

        self.__load_memusage()
        if os.path.isfile(f'{self.dirname}/{CHECKPOINTS_FILENAME}'):
            self.__load_checkpoints()



    def __load_memusage(self):
        memusage = np.nan_to_num(
            np.genfromtxt(
                f'{self.dirname}/{MEMUSAGE_FILENAME}',
                dtype=np.float64, delimiter='\t', skip_header=1,
            ),
            nan=0.0,
        )

        exec_start_timestamp: np.float64 = memusage[0,0]
        memusage[:,0] = memusage[:,0] - exec_start_timestamp

        if not (len(memusage.shape) == 2 and memusage.shape[1] == 2):
            raise ValueError("Got wrong data from the memusage table: memusage has to be a table (2d array) with 2 columns")

        self.memusage = memusage
        self.exec_time = memusage[-1,0]
        self.peakmemusage = memusage[:,1].max()
        self.exec_start_timestamp = exec_start_timestamp



    def __load_checkpoints(self):
        with open(f'{self.dirname}/{CHECKPOINTS_FILENAME}', newline='') as tsvfile:
            csvreader = csv.reader(tsvfile, delimiter='\t')
            for timestamp, comment in csvreader:
                timestamp_float = np.float64(timestamp) - self.exec_start_timestamp
                self.mark_timestamps.append(timestamp_float)
                self.mark_comment.append(comment)

                if comment.startswith('loaded'):
                    self.mark_marker.append('^')
                elif 'BiDiPBWT:' in comment:
                    if 'BiDiPBWT: 1 - ' in comment:
                        self.mark_marker.append('$1$')
                    elif 'BiDiPBWT: 2 - ' in comment:
                        self.mark_marker.append('$2$')
                    elif 'BiDiPBWT: 3 - ' in comment:
                        self.mark_marker.append('$3$')
                    elif 'BiDiPBWT: 4 - ' in comment:
                        self.mark_marker.append('$4$')
                    elif 'BiDiPBWT: 5 - ' in comment:
                        self.mark_marker.append('$5$')
                elif 'f-b' in comment:
                    if 'forward values' in comment:
                        self.mark_marker.append('$F$')
                    elif 'backward values' in comment:
                        self.mark_marker.append('$B$')
                        self.interpolation_sections_start.append(timestamp_float)
                        # print(f"{self.dirname}: sect start at {timestamp}")
                    elif 'interpolat' in comment:
                        self.mark_marker.append('$I$')
                        self.interpolation_sections_end.append(timestamp_float)
                        # print(f"{self.dirname}: sect end at {timestamp}")
                elif comment.startswith('hap0'):
                    self.mark_marker.append('o')
                elif comment.startswith('hap1'):
                    self.mark_marker.append('p')
                elif comment.startswith('saved'):
                    self.mark_marker.append('v')
                else:
                    self.mark_marker.append('.')



    def get_memusage_at_time(self, timestamp: np.float64):
        exe_time_start = self.memusage[0,0]
        exe_time_end = self.memusage[-1,0]

        if not exe_time_start <= timestamp <= exe_time_end:
            return np.float64(0)

        i = np.argmax(self.memusage[:,0] > timestamp)
        interval_start: np.float64      = self.memusage[i-1,0]
        interval_end: np.float64        = self.memusage[i,0]
        mem_interval_start: np.float64  = self.memusage[i-1,1]
        mem_interval_end: np.float64    = self.memusage[i,1]

        return (
            (timestamp-interval_start)/(interval_end-interval_start)
        ) * (mem_interval_end - mem_interval_start) + mem_interval_start






def plot_benchmark_logs(benchmarks: List[benchmark_logs], title: str, filename: Union[str,None] = None):
    benchmarks_memusage_converted: List[np.ndarray] = []
    for bm in benchmarks:
        memusage_converted = np.copy(bm.memusage)
        memusage_converted[:,0] = ms_to_min(memusage_converted[:,0])
        memusage_converted[:,1] = B_to_MB(memusage_converted[:,1])
        benchmarks_memusage_converted.append(memusage_converted)    

    xlim_right = np.float64(
        max([memusage[-1, 0] for memusage in benchmarks_memusage_converted]) # maximum execution time
    )
    ylim_top = np.float64(
        max([memusage[:, 1].max() for memusage in benchmarks_memusage_converted]) # maximum memory usage
    )
    ylim_top = ylim_top*1.2

    plt.ylim(bottom=0, top=ylim_top)
    plt.xlim(left=0, right=xlim_right)

    plt.xlabel('Execution time (minutes)')
    plt.ylabel('Memory usage (MB)')


    for i in range(len(benchmarks)):
        exec_time_minutes = np.round(ms_to_min(benchmarks[i].exec_time), 2)
        peakmem = np.round(B_to_MB(benchmarks[i].peakmemusage), 0)

        plt.plot(
            benchmarks_memusage_converted[i][:,0], benchmarks_memusage_converted[i][:,1],
            linewidth=1,
            color=benchmarks[i].color,
            label=f'{benchmarks[i].name}: '+
            f'{exec_time_minutes} min, {peakmem:.0f} MB'
        )

        memusages = [benchmarks[i].get_memusage_at_time(t) for t in benchmarks[i].mark_timestamps]

        converted_timestamps = [ms_to_min(t) for t in benchmarks[i].mark_timestamps]
        converted_memusages = [B_to_MB(m) for m in memusages]

        for t in range(len(benchmarks[i].mark_timestamps)):
            plt.scatter(
                converted_timestamps[t],
                converted_memusages[t],
                linewidths=2, edgecolors=benchmarks[i].color, color='#ffffff00', # type: ignore # str | None does work
                marker=benchmarks[i].mark_marker[t], # type: ignore
            )

    ax = plt.gca()
    ax.tick_params(color=(0, 0, 0, 0.3))
    for spine in ax.spines.values():
        spine.set_edgecolor((0, 0, 0, 0.3))

    plt.title(title)
    plt.legend()

    if filename:
        plt.savefig(filename, dpi=300)
        print(f"plot saved to the file:")
        print(filename)
    else:
        plt.show()



benchmarks = {
    # 0: benchmark_logs(
    #     name='v2022.07.28_very-1st-run',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20',
    #     color='#000000',
    # ),
    # 1: benchmark_logs(
    #     name='v2022.07.28 extended log',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_extended-log',
    #     color='#2faf3fbf',
    # ),
    # 2: benchmark_logs(
    #     name='v2022.07.28 opt1',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opt1',
    #     color='#af6f2fbf',
    # ),
    # 3: benchmark_logs(
    #     name='v2022.07.28 opts01-02',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opts01-02',
    #     color='#ff4f2fdf',
    # ),
    # 4: benchmark_logs(
    #     name='v2022.07.28 opts01-03',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opts01-03',
    #     color='#bf4fbfff',
    # ),
    # 5: benchmark_logs(
    #     name='v2022.07.28 opts01-03_b',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opts01-03_b',
    #     color='#8f7fbfff',
    # ),
    # 6: benchmark_logs(
    #     name='v2022.07.28 opts01-03_c',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opts01-03_c',
    #     color='#4f9fbfff',
    # ),
    # 7: benchmark_logs(
    #     name='v2022.07.28 opts01-03_d',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opts01-03_d',
    #     color='#2fcfbfff',
    # ),
    # 9: benchmark_logs(
    #     name='v2022.07.28 opts01-02_f',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opts01-02_f',
    #     color='#9f9f2fff',
    # ),
    # 8: benchmark_logs(
    #     name='v2022.07.28 opts01-03_e',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opts01-03_e',
    #     color='#2fcfbfff',
    # ),



    # 0: benchmark_logs(
    #     name='v2022.07.28',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20',
    #     color='#af6f2fbf',
    # ),
    # 2: benchmark_logs(
    #     name='v2022.07.28 opt1',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opt1',
    #     color='#2f3fbf',
    # ),


    # 2: benchmark_logs(
    #     name='v2022.07.28 opt1',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opt1',
    #     color='#af6f2fbf',
    # ),
    # 3: benchmark_logs(
    #     name='v2022.07.28 opts01-02',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opts01-02',
    #     color='#2f3fbf',
    # ),


    # 9: benchmark_logs(
    #     name='v2022.07.28 opts01-02_f',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opts01-02_f',
    #     color='#af6f2fbf',
    # ),
    # 8: benchmark_logs(
    #     name='v2022.07.28 opts01-03_e',
    #     dirname='memory-benchmark_2022.07.26-kukrefactored_chr20_opts01-03_e',
    #     color='#2f3fbf',
    # ),


    # 10: benchmark_logs(
    #     name='v2022.07.28',
    #     dirname='memory-benchmark_2022-07-26-kukrefactored_2022-08-22',
    #     color='#cf4f2fbf',
    # ),
    # 11: benchmark_logs(
    #     name='v2022.07.28_b',
    #     dirname='memory-benchmark_2022-07-26-kukrefactored_2022-08-22_b',
    #     color='#af6f2fbf',
    # ),
    # 12: benchmark_logs(
    #     name='v2022.07.28_c',
    #     dirname='memory-benchmark_2022-07-26-kukrefactored_2022-08-22_c',
    #     color='#8f8f2fbf',
    # ),
    # 13: benchmark_logs(
    #     name='v2022.07.28 opts01-04',
    #     dirname='memory-benchmark_opts01-04_2022-08-22',
    #     color='#2f1fdf',
    # ),
    # 14: benchmark_logs(
    #     name='v2022.07.28 opts01-04_b',
    #     dirname='memory-benchmark_opts01-04_2022-08-22_b',
    #     color='#1f3fbf',
    # ),
    # 15: benchmark_logs(
    #     name='v2022.07.28 opts01-04_c',
    #     dirname='memory-benchmark_opts01-04_2022-08-22_c',
    #     color='#0f5f9f',
    # ),

    # 12: benchmark_logs(
    #     name='v2022.07.28',
    #     dirname='memory-benchmark_2022-07-26-kukrefactored_2022-08-22_c',
    #     color='#af6f2fbf',
    # ),
    # 15: benchmark_logs(
    #     name='v2022.07.28 opts01-04',
    #     dirname='memory-benchmark_opts01-04_2022-08-22_c',
    #     color='#2f3fbf',
    # ),


    # 16: benchmark_logs(
    #     name='v2022.07.28',
    #     dirname='memory-benchmark_v2022.07.26-kukrefactored_2022-08-22-pack1',
    #     color='#ef2f2fbf',
    # ),
    # 17: benchmark_logs(
    #     name='v2022.07.28 opt1',
    #     dirname='memory-benchmark_opt01_2022-08-22-pack1',
    #     color='#cf4f2fbf',
    # ),
    # 18: benchmark_logs(
    #     name='v2022.07.28 opts01-02',
    #     dirname='memory-benchmark_opts01-02_2022-08-22-pack1',
    #     color='#af6f2fbf',
    # ),
    # 19: benchmark_logs(
    #     name='v2022.07.28 opts01-03',
    #     dirname='memory-benchmark_opts01-03_2022-08-22-pack1',
    #     color='#8f8f2fbf',
    # ),
    # 20: benchmark_logs(
    #     name='v2022.07.28 opts01-04',
    #     dirname='memory-benchmark_opts01-04_2022-08-22-pack1',
    #     color='#2f3fbf',
    # ),

    # 21: benchmark_logs(
    #     name='v2022.07.28 opts01-04_c',
    #     dirname='memory-benchmark_opts01-04_2022-08-22_c',
    #     color='#af6f2fbf',
    # ),
    # 22: benchmark_logs(
    #     name='v2022.07.28 opts01-04_dev1b',
    #     dirname='memory-benchmark_opts01-04_dev1_2022-08-31_2b',
    #     color='#2f3fbf',
    # ),
    # 22: benchmark_logs(
    #     name='v2022.07.28 opts01-04_dev1a',
    #     dirname='memory-benchmark_opts01-04_dev1_2022-08-31_2',
    #     color='#2f1fdf',
    # ),
    # 23: benchmark_logs(
    #     name='v2022.07.28 opts01-04_dev1b',
    #     dirname='memory-benchmark_opts01-04_dev1_2022-08-31_2b',
    #     color='#0f5f9f',
    # ),


    # 24: benchmark_logs(
    #     name='membench_opt05_2022-09-21_01',
    #     dirname='membench_opt05_2022-09-21_01',
    #     color='#2f3fbf',
    # ),
    # 25: benchmark_logs(
    #     name='membench_opt05_2022-09-21_02',
    #     dirname='membench_opt05_2022-09-21_02',
    #     color='#2f3fbf',
    # ),
    # 26: benchmark_logs(
    #     name='membench_opt05_2022-09-21_03',
    #     dirname='membench_opt05_2022-09-21_03',
    #     color='#2f3fbf',
    # ),
    # 27: benchmark_logs(
    #     name='membench_opt05_2022-09-21_04',
    #     dirname='membench_opt05_2022-09-21_04',
    #     color='#2f3fbf',
    # ),
    # 28: benchmark_logs(
    #     name='opt05_2022-09-21_05',
    #     dirname='membench_opt05_2022-09-21_05',
    #     color='#2f3fbf',
    # ),



    ######## 2000: benchmark_logs(
    ########     name='v2022.07.28 stage#1 optim',
    ########     dirname='memory-benchmark_opts01-04_2022-08-22_c',
    ########     color='#af6f2fbf',
    ######## ),
    ######## 2001: benchmark_logs(
    ########     name='v2022.07.28 stage#2 optim',
    ########     dirname='membench_opt05_2022-09-21_05',
    ########     color='#2f3fbf',
    ######## ),
    # 29: benchmark_logs(
    #     name='membench_opt05_2022-09-21_06',
    #     dirname='membench_opt05_2022-09-21_06',
    #     color='#2f3fbf',
    # ),


    # 30: benchmark_logs(
    #     name='v2022.07.28 stage#1 optim',
    #     dirname='membench_opts01-04_2022-10-02_3samples_01',
    #     color='#af6f2fbf',
    # ),
    # 31: benchmark_logs(
    #     name='v2022.07.28 stage#2 optim',
    #     dirname='membench_opts01-05_unrefactored_2022-10-02_3samples_01',
    #     color='#2f3fbf',
    # ),
    # 32: benchmark_logs(
    #     name='v2022.07.28 stage#1 optim',
    #     dirname='membench_opts01-04_2022-10-02_6samples_01',
    #     color='#af6f2fbf',
    # ),
    # 33: benchmark_logs(
    #     name='v2022.07.28 stage#2 optim',
    #     dirname='membench_opts01-05_unrefactored_2022-10-02_6samples_01',
    #     color='#2f3fbf',
    # ),

    # 34: benchmark_logs(
    #     name='v2022.07.28 stage#2 optim',
    #     dirname='membench_stage2_halfrefactored_2022-10-28_chr1_1sample',
    #     color='#2f5fbf',
    # ),
    # 35: benchmark_logs(
    #     name='v2022.07.28 stage#2 optim (B)',
    #     dirname='membench_stage2_halfrefactored_2022-10-28_chr1_1sample_02',
    #     color='#4f2fbf',
    # ),


    # 36: benchmark_logs(
    #     name='membench_opt05_2022-09-21_05',
    #     dirname='membench_opt05_2022-09-21_05',
    #     color='#2f5fbf',
    # ),

    # 37: benchmark_logs(
    #     name='membench_stage2_refactored_2022-10-31_1sample',
    #     dirname='membench_stage2_refactored_2022-10-31_1sample',
    #     color='#4f2fbf',
    # ),
    # 38: benchmark_logs(
    #     name='membench_stage2_refactored_2022-10-31_1sample_02',
    #     dirname='membench_stage2_refactored_2022-10-31_1sample_02',
    #     color='#6f1fbf',
    # ),
    # 39: benchmark_logs(
    #     name='membench_stage2_refactored_2022-10-31_1sample_03 (interpolate_parallel_packed_prev)',
    #     dirname='membench_stage2_refactored_2022-10-31_1sample_03',
    #     color='#6f1fbf',
    # ),
    # 40: benchmark_logs(
    #     name='membench_stage2_refactored_2022-10-31_1sample_04 (- cnvrsn of hapld indxs)',
    #     dirname='membench_stage2_refactored_2022-10-31_1sample_04',
    #     color='#7f0fbf',
    # ),
    # 41: benchmark_logs(
    #     name='membench_stage2_refactored_2022-10-31_1sample_05 (added 2 `del`s)',
    #     dirname='membench_stage2_refactored_2022-10-31_1sample_05',
    #     color='#9f0fbf',
    # ),
    # 42: benchmark_logs(
    #     name='membench_stage2_refactored_2022-10-31_1sample_06 (wrap impttn loop in func)',
    #     dirname='membench_stage2_refactored_2022-10-31_1sample_06',
    #     color='#af07bf',
    # ),
    # 43: benchmark_logs(
    #     name='membench_stage2_refactored_2022-10-31_1sample_07 (added 1 `del`)',
    #     dirname='membench_stage2_refactored_2022-10-31_1sample_07',
    #     color='#b703bf',
    # ),
    # 44: benchmark_logs(
    #     name='membench_stage2_refactored_2022-10-31_1sample_08 (slightly chngd readlines)',
    #     dirname='membench_stage2_refactored_2022-10-31_1sample_08',
    #     color='#b703bf',
    # ),
    # 45: benchmark_logs(
    #     name='membench_stage2_dev1.5.1_2022-10-31_1sample_01',
    #     dirname='../selphi_fdraft_restored_dev1.5.1/membench_stage2_dev1.5.1_2022-10-31_1sample_01',
    #     color='#b703bf',
    # ),
    # 46: benchmark_logs(
    #     name='membench_stage2_dev1.5.0_2022-10-31_1sample_01',
    #     dirname='../selphi_fdraft_restored_dev1.5.0/membench_stage2_dev1.5.0_2022-10-31_1sample_01',
    #     color='#b703bf',
    # ),
    # 47: benchmark_logs(
    #     name='membench_stage2_opt05_B_2022-10-31_1sample_01',
    #     dirname='../selphi_fdraft_restored_opt05_B/membench_stage2_opt05_B_2022-10-31_1sample_01',
    #     color='#b703bf',
    # ),
    # 48: benchmark_logs(
    #     name='membench_stage2_dev1.3.1_2022-10-31_1sample_01',
    #     dirname='../selphi_fdraft_restored_dev1.3.1/membench_stage2_dev1.3.1_2022-10-31_1sample_01',
    #     color='#b703bf',
    # ),
    # 49: benchmark_logs(
    #     name='membench_stage2_dev1.3.2_2022-10-31_1sample_01',
    #     dirname='../selphi_fdraft_restored_dev1.3.2/membench_stage2_dev1.3.2_2022-10-31_1sample_01',
    #     color='#b703bf',
    # ),
    # 50: benchmark_logs(
    #     name='membench_stage2_refactored_ibds-extrctr_2022-11-01_1sample_01',
    #     dirname='membench_stage2_refactored_ibds-extrctr_2022-11-01_1sample_01',
    #     color='#b703bf',
    # ),
    # 51: benchmark_logs(
    #     name='membench_stage2_refactored_ibds-extrctr-sep14-failed_2022-11-01_1sample_01',
    #     dirname='membench_stage2_refactored_ibds-extrctr-sep14-failed_2022-11-01_1sample_01',
    #     color='#b703bf',
    # ),
    # 52: benchmark_logs(
    #     name='membench_stage2_refactored_ibds-extrctr-sep14_2022-11-01_1sample_01',
    #     dirname='membench_stage2_refactored_ibds-extrctr-sep14_2022-11-01_1sample_01',
    #     color='#b703bf',
    # ),
    # 53: benchmark_logs(
    #     name='membench_stage2_dev1.3.1_rfctrd-clib_2022-10-31_1sample_01',
    #     dirname='../selphi_fdraft_restored_dev1.3.1/membench_stage2_dev1.3.1_rfctrd-clib_2022-10-31_1sample_01',
    #     color='#b703bf',
    # ),
    # 54: benchmark_logs(
    #     name='membench_stage2_dev1.3.2_2022-10-31_1sample_02',
    #     dirname='../selphi_fdraft_restored_dev1.3.1_vs_dev1.3.2/membench_stage2_dev1.3.2_2022-10-31_1sample_02',
    #     color='#b703bf',
    # ),
    # 55: benchmark_logs(
    #     name='membench_stage2_dev1.3.1_2022-10-31_1sample_02',
    #     dirname='../selphi_fdraft_restored_dev1.3.1_vs_dev1.3.2/membench_stage2_dev1.3.1_2022-10-31_1sample_02',
    #     color='#b703bf',
    # ),
    # 56: benchmark_logs(
    #     name='membench_stage2_dev1.3.2-fixA_2022-10-31_1sample_01',
    #     dirname='../selphi_fdraft_restored_dev1.3.1_vs_dev1.3.2/membench_stage2_dev1.3.2-fixA_2022-10-31_1sample_01',
    #     color='#b703bf',
    # ),
    # 57: benchmark_logs(
    #     name='membench_stage2_refactored_ibds-extrctr-sep14-utils-fix_2022-11-01_1sample_01',
    #     dirname='membench_stage2_refactored_ibds-extrctr-sep14-utils-fix_2022-11-01_1sample_01',
    #     color='#b703bf',
    # ),
    # 58: benchmark_logs(
    #     name='membench_stage2_refactored_utils-fix_2022-11-01_1sample_01',
    #     dirname='membench_stage2_refactored_utils-fix_2022-11-01_1sample_01',
    #     color='#4f2fbf',
    # ),
    # 59: benchmark_logs(
    #     name='membench_stage2_refactored_utils-fix_2022-11-01_1sample_02',
    #     dirname='membench_stage2_refactored_utils-fix_2022-11-01_1sample_02',
    #     color='#4f2fbf',
    # ),
    # 60: benchmark_logs(
    #     name='membench_stage2_refactored_utils-fix_2022-11-01_1sample_03',
    #     dirname='membench_stage2_refactored_utils-fix_2022-11-01_1sample_03',
    #     color='#4f2fbf',
    # ),

    # 61: benchmark_logs(
    #     name='membench_stage2_refactored_utils-fix_2022-11-01_1sample_04',
    #     dirname='membench_stage2_refactored_utils-fix_2022-11-01_1sample_04',
    #     color='#4f2fbf',
    # ),
    # 61: benchmark_logs(
    #     name='membench_stage2_refactored_ibds-extrctr_2022-11-01_1sample_02',
    #     dirname='membench_stage2_refactored_ibds-extrctr_2022-11-01_1sample_02',
    #     color='#b703bf',
    # ),


    # 2100: benchmark_logs(
    #     name='v2022.07.28 stage#2 optim',
    #     dirname='membench_opt05_2022-09-21_05',
    #     color='#2f5fbf',
    # ),
    # 2101: benchmark_logs(
    #     name='v2022.07.28 stage#2 optim (refactored)',
    #     dirname='membench_stage2_refactored_utils-fix_2022-11-01_1sample_04',
    #     color='#4f2fbf',
    # ),


    # 62: benchmark_logs(
    #     name='stage2_refactored_2022-11-10_chr1_02',
    #     dirname='membench_stage2_refactored_2022-11-10_chr1_02',
    #     color='#2f3fbf',
    # ),


    # 63: benchmark_logs(
    #     name='stage2rfct_ibdsx-sep14-rfct01_2022-11-12_1smpl_01',
    #     dirname='membench_stage2_refactored_ibds-extrctr-sep14-refact01_2022-11-12_1sample_01',
    #     color='#4f2fbf',
    # ),

    # 64: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_test01',
    #     dirname='membench_stage3-dev_2022-11-17_test01',
    #     color='#2f3fbf',
    # ),

    # 65: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06',
    #     dirname='membench_stage3-dev_2022-11-17_opt06',
    #     color='#6f1fbf',
    # ),
    #####66: benchmark_logs(
    #####    name='membench_stage3-dev_2022-11-17_opt06-07_02',
    #####    dirname='membench_stage3-dev_2022-11-17_opt06-07_02',
    #####    color='#b703bf',
    #####),


    # 67: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-08_01',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08_01',
    #     color='#ef035f',
    # ),
    # 68: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-08_02',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08_02',
    #     color='#efa32f',
    # ),
    # 69: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-08_03',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08_03',
    #     color='#3ff32f',
    # ),

    # 70: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-09_01',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-09_01',
    #     color='#0faf8f',
    # ),
    # 71: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-09_02',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-09_02',
    #     color='#0faf8f',
    # ),
    # 72: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-09_03',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-09_03',
    #     color='#0faf8f',
    # ),
    # 73: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-09_04',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-09_04',
    #     color='#0faf8f',
    # ),
    # 74: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-09_05',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-09_05',
    #     color='#0faf8f',
    # ),
    # 75: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-09_06',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-09_06',
    #     color='#0faf8f',
    # ),
    ### 76: benchmark_logs(
    ###     name='membench_stage3-dev_2022-11-17_opt06-08_04',
    ###     dirname='membench_stage3-dev_2022-11-17_opt06-08_04',
    ###     color='#0f8f3f',
    ### ),
    ### 77: benchmark_logs(
    ###     name='membench_stage3-dev_2022-11-17_opt06-08_05',
    ###     dirname='membench_stage3-dev_2022-11-17_opt06-08_05',
    ###     color='#0faf8f',
    ### ),
    # 78: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-08_05B',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08_05B',
    #     color='#0f2fff',
    # ),
    ####79: benchmark_logs(
    ####    name='membench_stage3-dev_2022-11-17_opt06-08_05D',
    ####    dirname='membench_stage3-dev_2022-11-17_opt06-08_05D',
    ####    color='#0f2fff',
    ####),
    # 2001: benchmark_logs(
    #     name='v2022.07.28 stage#2 optim',
    #     dirname='membench_opt05_2022-09-21_05',
    #     color='#af6f2fbf',
    # ),
    # 80: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-08_06',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08_06',
    #     color='#6f1fbf',
    # ),
    # 81: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-08_06B',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08_06B',
    #     color='#2f0f6f',
    # ),
    # 82: benchmark_logs( # same as opt06-08_06
    #     name='membench_stage3-dev_2022-11-17_opt06-08_07',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08_07',
    #     color='#6f1fbf',
    # ),
    # 83: benchmark_logs( # same as opt06-08_06
    #     name='membench_stage3-dev_2022-11-17_opt06-08_07B',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08_07B', 
    #     color='#6f1fbf',
    # ),
    # 84: benchmark_logs(  # as opt06-08_06, but div_matrix is not precalculated fully, but for each cv under the loop 
    #     name='membench_stage3-dev_2022-11-17_opt06-08_08',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08_08',
    #     color='#2f0f6f',
    # ),
    # 85: benchmark_logs(  # as opt06-08_06, but div_matrix is not precalculated fully, but for each cv under the loop 
    #     name='membench_stage3-dev_2022-11-17_opt06-08_08B',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08_08B',
    #     color='#2f0f6f',
    # ),
    # 86: benchmark_logs(  # as opt06-08_06, but div_matrix is not precalculated fully, but for each cv under the loop 
    #     name='membench_stage3-dev_2022-11-17_opt06-08_08C', # C is different cuz memusage is now every 0.2 sec vs 0.3 before
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08_08C',
    #     color='#2f0f6f',
    # ),

    ####87: benchmark_logs(  # as opt06-08_06, but div_matrix is not precalculated fully, but for each cv under the loop 
    ####    name='membench_stage3-dev_2022-11-17_opt06-08,10_01',
    ####    dirname='membench_stage3-dev_2022-11-17_opt06-08,10_01',
    ####    color='#b703bf',
    ####),
    # 88: benchmark_logs(  # as opt06-10_01, but this is with njit(fastmath=True) for all BInBJ funcs
    #     name='membench_stage3-dev_2022-11-17_opt06-08,10_02',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08,10_02',
    #     color='#ef035f',
    # ),
    # 89: benchmark_logs(  # as opt06-10_02, but BiDiPBWT is combined with create_composite_ref_panel under numba
    #     name='membench_stage3-dev_2022-11-17_opt06-08,10_03',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08,10_03',
    #     color='#ff330f',
    # ),
    # 90: benchmark_logs(  # as opt06-10_02, but BiDiPBWT calculates BI and BJ separetly via the same function
    #     name='membench_stage3-dev_2022-11-17_opt06-08,10_04',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08,10_04',
    #     color='#ff330f',
    # ),
    # 91: benchmark_logs(  # as opt06-10_01, but this is with njit(fastmath=True) for all BInBJ funcs
    #     name='membench_stage3-dev_2022-11-17_opt06-08,10_02B',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08,10_02B',
    #     color='#ef035f',
    # ),
    # 92: benchmark_logs(  # as opt06-10_02, but vectorized calculation of the initial ref panel
    #     name='membench_stage3-dev_2022-11-17_opt06-08,10_05',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08,10_05',
    #     color='#af830f',
    # ),
    # 93: benchmark_logs(  # as opt06-10_02, but vectorized+njitted calculation of the initial ref panel
    #     name='membench_stage3-dev_2022-11-17_opt06-08,10_06',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08,10_06',
    #     color='#af830f',
    # ),
    # 94: benchmark_logs(  # as opt06-10_02, but removed redundant arrays in-between filters calculations
    #     name='membench_stage3-dev_2022-11-17_opt06-08,10_07',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08,10_07',
    #     color='#af830f',
    # ),
    # 95: benchmark_logs(
    #     name='membench_stage3-dev_2022-11-17_opt06-08,10_02C',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08,10_02C',
    #     color='#af830f',
    # ),
    # 96: benchmark_logs(  # another attempt of what did in opt06-10_07
    #     name='membench_stage3-dev_2022-11-17_opt06-08,10_08',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08,10_08',
    #     color='#aff30f',
    # ),
    # 97: benchmark_logs(  # as opt06-10_02, but a bit of 07, + njitted forward and backward algos
    #     name='membench_stage3-dev_2022-11-17_opt06-08,10_09',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08,10_09',
    #     color='#aff30f',
    # ),
    # 98: benchmark_logs(  # as opt06-10_02, but a bit of 07
    #     name='membench_stage3-dev_2022-11-17_opt06-08,10_02E',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08,10_02E',
    #     color='#2fa30f',
    # ),
    # 99: benchmark_logs(  # as opt06-10_02, but a bit of 07
    #     name='membench_stage3-dev_2022-11-17_opt06-08,10_02F',
    #     dirname='membench_stage3-dev_2022-11-17_opt06-08,10_02F',
    #     color='#2fa30f',
    # ),



    3000: benchmark_logs(
        name='v2022.07.28 stage#2 optim',
        dirname='membench_opt05_2022-09-21_05',
        color='#af6f2fbf',
    ),
    3001: benchmark_logs(  # as opt06-10_01, but this is with njit(fastmath=True) for all BInBJ funcs
        name='v2022.07.28 stage#3 optim',
        dirname='membench_stage3-dev_2022-11-17_opt06-08,10_02B',
        color='#2f3fbf',
    ),




}




def memusage_AUC(benchmark: benchmark_logs, time_interval: Union[Tuple[np.float64, np.float64], None] = None):
    T = 0 # time
    M = 1 # memory

    memusage = benchmark.memusage

    if time_interval is not None:
        start_T, start_M = time_interval[0], benchmark.get_memusage_at_time(time_interval[0])
        end_T,   end_M   = time_interval[1], benchmark.get_memusage_at_time(time_interval[1])

        start_T_next_idx = np.argmax(memusage[:,T] >= start_T)
        end_T_next_idx   = np.argmax(memusage[:,T] >= end_T)

        interval_memusage = np.concatenate(
            [
                np.array([[start_T, start_M]]),
                memusage[start_T_next_idx:end_T_next_idx],
                np.array([[end_T, end_M]]),
            ],
            axis=0
        )

        memusage_i = interval_memusage[:-1]   # elements #i
        memusage_ip1 = interval_memusage[1:]  # elements #i+1

    else:
        memusage_i = memusage[:-1]   # elements #i
        memusage_ip1 = memusage[1:]  # elements #i+1

    """
    formula for area pieces:
    a0 = ((m1+m0)/2)*(t1-t0)
    """
    AUC = (
        ((memusage_ip1[:,M] + memusage_i[:,M]) / 2) * (memusage_ip1[:,T] - memusage_i[:,T])
    ).sum()
    return AUC



# stage0_optim_1sample_AUC = memusage_AUC(benchmarks[12])
# stage1_optim_1sample_AUC = memusage_AUC(benchmarks[15])

# print(f"stage1vs0 opt 1sample AUC opt ratio: {((stage1_optim_1sample_AUC/stage0_optim_1sample_AUC))}")

# print(f"{B_ms_to_GB_min(memusage_AUC(benchmarks[34]))=}")
# print(f"{B_ms_to_GB_min(memusage_AUC(benchmarks[35]))=}")


# stage1_optim_1sample_AUC = memusage_AUC(benchmarks[2000])
# stage2_optim_1sample_AUC = memusage_AUC(benchmarks[2001])
# print(f"{B_ms_to_GB_min(memusage_AUC(benchmarks[2000]))=}")
# print(f"{B_ms_to_GB_min(memusage_AUC(benchmarks[2001]))=}")
# print(f"stage2vs1 opt 1sample AUC opt ratio: {((stage2_optim_1sample_AUC/stage1_optim_1sample_AUC))}")

# # stage1_optim_3samples_AUC = memusage_AUC(benchmarks[30])
# # stage2_optim_3samples_AUC = memusage_AUC(benchmarks[31])
# # print(f"stage2vs1 opt 3samples AUC opt ratio: {((stage2_optim_3samples_AUC/stage1_optim_3samples_AUC))}")

# # stage1_optim_6samples_AUC = memusage_AUC(benchmarks[32])
# # stage2_optim_6samples_AUC = memusage_AUC(benchmarks[33])
# # print(f"stage2vs1 opt 6samples AUC opt ratio: {((stage2_optim_6samples_AUC/stage1_optim_6samples_AUC))}")

# # print(f"stage2 opt 3samples vs 1sample AUC ratio: {((stage2_optim_3samples_AUC/stage2_optim_1sample_AUC))}")
# # print(f"stage2 opt 6samples vs 1sample AUC ratio: {((stage2_optim_6samples_AUC/stage2_optim_1sample_AUC))}")




# def memusage_AUC_interpolation8cores(benchmark: benchmark_logs):
#     if not len(benchmark.interpolation_sections_start) == len(benchmark.interpolation_sections_end):
#         raise ValueError(f"unable to calculate memusage_AUC_interpolation8cores for this benchmark: interpolation interval is defined to be between where a backward algo ends and where the interpolation ends. But this benchmark doesn't have an equal number of backward algo endings ({benchmark.interpolation_sections_start}) and interpolation endings ({benchmark.interpolation_sections_end})")


#     AUC_from_absolute_time = memusage_AUC(benchmark)

#     AUC_7otherCoresTime = 0
#     for i in range(len(benchmark.interpolation_sections_start)):
#         AUC_7otherCoresTime += memusage_AUC(
#             benchmark,
#             time_interval=(benchmark.interpolation_sections_start[i], benchmark.interpolation_sections_end[i])
#         )

#     AUC = AUC_from_absolute_time + 7*AUC_7otherCoresTime

#     return AUC

# def memusage_AUC_interpolation4cores(benchmark: benchmark_logs):
#     if not len(benchmark.interpolation_sections_start) == len(benchmark.interpolation_sections_end):
#         raise ValueError(f"unable to calculate memusage_AUC_interpolation8cores for this benchmark: interpolation interval is defined to be between where a backward algo ends and where the interpolation ends. But this benchmark doesn't have an equal number of backward algo endings ({benchmark.interpolation_sections_start}) and interpolation endings ({benchmark.interpolation_sections_end})")


#     AUC_from_absolute_time = memusage_AUC(benchmark)

#     AUC_4otherCoresTime = 0
#     for i in range(len(benchmark.interpolation_sections_start)):
#         AUC_4otherCoresTime += memusage_AUC(
#             benchmark,
#             time_interval=(benchmark.interpolation_sections_start[i], benchmark.interpolation_sections_end[i])
#         )

#     AUC = AUC_from_absolute_time + 3*AUC_4otherCoresTime

#     return AUC


# # print(f"stage#1 1sample memusage_AUC_interpolation8cores: {memusage_AUC_interpolation8cores(benchmarks[2000])}")
# stage1_optim_1sample_interpolation8cores_AUC = memusage_AUC_interpolation8cores(benchmarks[2000])
# print(f"{B_ms_to_GB_min(memusage_AUC_interpolation8cores(benchmarks[2000]))=}")
# print(f"{B_ms_to_GB_min(memusage_AUC_interpolation4cores(benchmarks[2000]))=}")
# print(f"stage2vs1 opt 1sample AUC opt ratio (given stage1 used 8 cores during interpolation): {((stage2_optim_1sample_AUC/stage1_optim_1sample_interpolation8cores_AUC))}")
# print(f"stage2 opt 1sample AUC: {stage2_optim_1sample_AUC}")



for i in benchmarks.keys():
    print(f"AUC({benchmarks[i].name}) = {round(B_ms_to_GB_min(memusage_AUC(benchmarks[i])), 3)} GB*min")

plot_benchmark_logs(
    [b for b in benchmarks.values()],
    title='selphi chr20',
    # filename="membench_opt05_2022-09-21_05b_show.png",
    # filename="benchmarks_plot_stage2refactored_ibds-extrctr-sep14-utils-fix_2022-11-01_chr20_01.png",
    # filename="benchmarks_plot_stage2refactored_utils-fix_2022-11-01_chr20_02.png",
    # filename="benchmarks_plot_stage2_dev1.3.1_rfctrd-clib_2022-11-01_chr20_01.png",
    # filename="benchmarks_plot_stage2_dev1.3.2-fixA_2022-11-01_chr20_01.png",
    # filename="benchmarks_plot_stage2refactored_utils-fix_2022-11-01_chr20_04_show.png",
    # filename="benchmarks_plot_stage2refactored_ibds-extrctr-utils-fix_2022-11-01_chr20_01.png",
    # filename="benchmarks_plot_stage2_chr1_02.png",
    # filename="benchplot_stage3-dev_2022-11-17_test01.png",

    filename="benchplot_stage3-dev_2022-11-17_opt06-08,10_02B_show.png",
)

