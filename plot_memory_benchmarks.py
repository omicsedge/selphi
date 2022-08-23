from typing import List, Union
import numpy as np
import csv

import matplotlib.pyplot as plt



CHECKPOINTS_FILENAME = 'checkpoints.tsv'
MEMUSAGE_FILENAME = 'memusage.tsv'

CHROM = 20


def b_to_MB(bytes: Union[int, float, np.float64, np.ndarray]):
    return bytes / 1_000_000

def ms_to_minute(milliseconds: Union[int, float, np.float64, np.ndarray]):
    return milliseconds / 60_000

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


    def __init__(self, name: str, dirname: str, color: Union[str, None] = None):
        self.mark_timestamps: List[np.float64] = []
        self.mark_comment: List[str] = []
        self.mark_marker: List[str] = []

        self.name = name
        self.dirname = dirname
        self.color = color

        self.__load_memusage()
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
                elif 'f-b & imputation:' in comment:
                    if 'forward values' in comment:
                        self.mark_marker.append('$F$')
                    elif 'backward values' in comment:
                        self.mark_marker.append('$B$')
                    elif 'interpolat' in comment:
                        self.mark_marker.append('$I$')
                elif comment.startswith('hap0'):
                    self.mark_marker.append('o')
                elif comment.startswith('hap1'):
                    self.mark_marker.append('p')
                elif comment.startswith('saved'):
                    self.mark_marker.append('v')
                else:
                    self.mark_marker.append('')



    def get_memusage_at_time(self, timestamp: np.float64):
        exe_time_start = self.memusage[0,0]
        exe_time_end = self.memusage[-1,0]

        if not exe_time_start <= timestamp < exe_time_end:
            return np.float64(0)

        i = np.argmax(self.memusage[:,0] > timestamp)
        interval_start: np.float64      = self.memusage[i-1,0]
        interval_end: np.float64        = self.memusage[i,0]
        mem_interval_start: np.float64  = self.memusage[i-1,1]
        mem_interval_end: np.float64    = self.memusage[i,1]

        return (
            (timestamp-interval_start)/(interval_end-interval_start)
        ) * (mem_interval_end - mem_interval_start) + mem_interval_start






def plot_benchmark_logs(benchmarks: List[benchmark_logs], title: str, filename: str):
    benchmarks_memusage_converted: List[np.ndarray] = []
    for bm in benchmarks:
        memusage_converted = np.copy(bm.memusage)
        memusage_converted[:,0] = ms_to_minute(memusage_converted[:,0])
        memusage_converted[:,1] = b_to_MB(memusage_converted[:,1])
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
        exec_time_minutes = np.round(ms_to_minute(benchmarks[i].exec_time), 2)
        peakmem = np.round(b_to_MB(benchmarks[i].peakmemusage), 0)

        plt.plot(
            benchmarks_memusage_converted[i][:,0], benchmarks_memusage_converted[i][:,1],
            linewidth=1,
            color=benchmarks[i].color,
            label=f'{benchmarks[i].name}: '+
            f'{exec_time_minutes} min, {peakmem:.0f} MB'
        )

        memusages = [benchmarks[i].get_memusage_at_time(t) for t in benchmarks[i].mark_timestamps]

        converted_timestamps = [ms_to_minute(t) for t in benchmarks[i].mark_timestamps]
        converted_memusages = [b_to_MB(m) for m in memusages]

        for t in range(len(benchmarks[i].mark_timestamps)):
            plt.scatter(
                converted_timestamps[t],
                converted_memusages[t],
                linewidths=2, edgecolors=benchmarks[i].color, color='#ffffff00',
                marker=benchmarks[i].mark_marker[t], # type: ignore
            )

    ax = plt.gca()
    ax.tick_params(color=(0, 0, 0, 0.3))
    for spine in ax.spines.values():
        spine.set_edgecolor((0, 0, 0, 0.3))

    plt.title(title)
    plt.legend()

    plt.savefig(filename, dpi=300)



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

    12: benchmark_logs(
        name='v2022.07.28',
        dirname='memory-benchmark_2022-07-26-kukrefactored_2022-08-22_c',
        color='#af6f2fbf',
    ),
    15: benchmark_logs(
        name='v2022.07.28 opts01-04',
        dirname='memory-benchmark_opts01-04_2022-08-22_c',
        color='#2f3fbf',
    ),


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
}

plot_benchmark_logs(
    [b for b in benchmarks.values()],
    title='selphi chr20',
    filename="benchmarks_plot_opts01-04_show.png",
)

