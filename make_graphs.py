import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pathlib
import seaborn as sns

from simple_term_menu import TerminalMenu

class StatisticsFileLoader:
    def __init__(self, stats_path = '/var/lib/sparse/stats'):
        self.stats_path = stats_path

    def select_from_options(self, options, title):
        terminal_menu = TerminalMenu(options, title=title)
        menu_entry_index = terminal_menu.show()
        if menu_entry_index is None:
            return None
        print(f"{title} {options[menu_entry_index]}")
        return options[menu_entry_index]

    def load_dataframe(self):
        filename = self.select_from_options(os.listdir(self.stats_path), "Dataframe file name:")
        if filename is None:
            return None
        filepath = os.path.join(self.stats_path, filename)

        try:
            return pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"File '{filename}' was not found in directory '{STATS_PATH}'. Make sure that it exists and is readable.")

    def parse_boxplot_frame(self):
        df = self.load_dataframe()

        if df is None:
            return None

        no_datasources = input("Number of data sources in experiment: ")
        used_scheduling = self.select_from_options(["OTN", "None"], "Scheduling method:")
        df = df.assign(Datasources=int(no_datasources), Scheduling=used_scheduling)

        # Statistics only for offloaded tasks
        df = df.loc[df['request_op']=='offload_task']

        return df

class StatisticsGraphPlotter:
    def __init__(self, write_path = '/var/lib/sparse/stats'):
        self.file_loader = StatisticsFileLoader()
        self.write_path = write_path

    def count_offload_task_statistics(self, df):
        """Calculates latencies from a Dataframe with time stamps, and returns a new Dataframe with the results.
        """
        df = df.loc[df['request_op']=='offload_task']

        # Drop first rows to ignore 'slow start'
        # (See e.g: https://stackoverflow.com/questions/64420777/opencv-cuda-api-very-slow-at-the-first-call)
        df.drop(index=df.index[:64], inplace=True)

        df['e2e_latency'] = df['response_received_at'] - df['processing_started_at']
        df['offload_latency'] = df['response_received_at'] - df['request_sent_at']

        # Scale latencies to milliseconds
        df['e2e_latency'] = df['e2e_latency'].apply(lambda x: 1000.0*x)
        df['offload_latency'] = df['offload_latency'].apply(lambda x: 1000.0*x)

        df = df.rename(columns={ 'e2e_latency': 'Latency (ms)', 'offload_latency': 'Offload latency (ms)' })

        return df

    def print_statistics(self):
        df = self.file_loader.load_dataframe()

        # Calculate statistics for offloaded tasks
        stats = self.count_offload_task_statistics(df)

        # Print statistics
        print(stats[['Latency (ms)', 'Offload latency (ms)']].describe())

    def plot_boxplot(self):
        plt.figure(figsize=(8,4))

        frames = []
        while True:
            df = self.file_loader.parse_boxplot_frame()
            if df is None:
                break
            frames.append(self.count_offload_task_statistics(df))

        ax = sns.boxplot(x="Datasources", y="Offload latency (ms)", hue="Scheduling", data=pd.concat(frames), palette="dark:grey")

        filepath = os.path.join(self.write_path, "scheduling_boxplot.png")
        plt.savefig(filepath, dpi=400)
        print(f"Saved boxplot to '{filepath}'")

if __name__ == "__main__":
    StatisticsGraphPlotter().plot_boxplot()
