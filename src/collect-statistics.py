#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from datetime import datetime
import os

from jtop import jtop
import pandas as pd

if __name__ == "__main__":
    print("")

    # Collect statistics
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
    OUTPUT_DIR = './data/stats'
    experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # GPU utilization
    gpu_filepath = os.path.join(OUTPUT_DIR, 'gpu-usage-{:s}.csv'.format(experiment_time))
    stats.to_csv(gpu_filepath, columns=['timestamp', 'GPU'], index=False)
    print("Saved GPU usage statistics to file '{:s}'".format(gpu_filepath))

    # RAM usage
    ram_filepath = os.path.join(OUTPUT_DIR, 'ram-usage-{:s}.csv'.format(experiment_time))
    stats.to_csv(ram_filepath, columns=['timestamp', 'RAM'], index=False)
    print("Saved RAM usage statistics to file '{:s}'".format(ram_filepath))

# EOF
