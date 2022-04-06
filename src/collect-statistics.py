#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from datetime import datetime
import importlib
import psutil
import time
import os

import pandas as pd

def collect_statistics_jtop():
    stats = start_time = None
    print("Collecting device statistics")
    with jtop() as jetson:
        # jetson.ok() will provide the proper update frequency
        while jetson.ok():
            try:
                # Read tegra stats
                new_frame = pd.DataFrame(jetson.stats, index=[jetson.stats['time']])

                # Replace datetime with timestamp (in seconds)
                timestamp = pd.Timestamp(new_frame['time'][0]).timestamp()

                # Replace RAM usage with relative value
                new_frame['RAM'] = new_frame['RAM'][0] / 2000000.0 * 100.0

                # Drop unnecessary columns
                new_frame = new_frame.drop(columns=['uptime', 'jetson_clocks', 'nvp model', 'NVENC', 'NVDEC', 'NVJPG'])
                if stats is None:
                    stats = new_frame
                    start_time = timestamp
                    new_frame['timestamp'] = 0
                else:
                    new_frame['timestamp'] = timestamp - start_time
                    stats = pd.concat([stats, new_frame])
                print(new_frame)

            except KeyboardInterrupt:
                break

    print("Stopped collecting device statistics")
    print("")

    # Print statistics
    print("Collected statistics")
    print(stats)

    # Store statistics
    print("")
    print("Saving collected statistics")

    # GPU utilization
    gpu_filepath = os.path.join(OUTPUT_DIR, 'gpu-usage-{:s}.csv'.format(experiment_time))
    stats.to_csv(gpu_filepath, columns=['timestamp', 'GPU'], index=False)
    print("Saved GPU usage statistics to file '{:s}'".format(gpu_filepath))

    # RAM usage
    ram_filepath = os.path.join(OUTPUT_DIR, 'ram-usage-{:s}.csv'.format(experiment_time))
    stats.to_csv(ram_filepath, columns=['timestamp', 'RAM'], index=False)
    print("Saved RAM usage statistics to file '{:s}'".format(ram_filepath))

    return stats

def collect_statistics_general(filepath, if_name = 'lo'):
    print(f'Writing statistics to file {filepath!r}')
    with open(filepath, 'a') as f:
        header_row = 'timestamp,bytes_sent,bytes_recv'
        print(header_row)
        f.write(header_row + '\n')

        nic_stats = psutil.net_io_counters(pernic=True)
        initial_time = initial_bytes_sent = initial_bytes_recv = None

        while True:
            try:
                # Read new measurements
                curr_time = time.time_ns()
                bytes_sent, bytes_recv = read_network_stats()
                if initial_time == None:
                    initial_time = curr_time
                    initial_bytes_sent = bytes_sent
                    initial_bytes_recv = bytes_recv
                    row = f"0,0,0"
                else:
                    row = f"{curr_time-initial_time},{bytes_sent-initial_bytes_sent},{bytes_recv-initial_bytes_recv}"

                # Write measurements
                print(row)
                f.write(row + '\n')
                time.sleep(1)

            except KeyboardInterrupt:
                break


def read_network_stats(if_name = 'lo'):
    nic_stats = psutil.net_io_counters(pernic=True)
    [bytes_sent, bytes_recv, packets_sent, packets_recv, errin, errout, dropin, dropout] = nic_stats[if_name]
    return bytes_sent, bytes_recv

if __name__ == "__main__":
    OUTPUT_DIR = './data/stats'
    experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(OUTPUT_DIR, f'statistics-{experiment_time}.csv')

    print("")

    if importlib.util.find_spec("jtop") is not None:
        from jtop import jtop

        collect_statistics_jtop()
    else:
        print("No jtop available, collecting general statistics")
        collect_statistics_general(filepath)

