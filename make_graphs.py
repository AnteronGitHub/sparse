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

    def select_from_options(self, options):
        terminal_menu = TerminalMenu(options)
        menu_entry_index = terminal_menu.show()
        if menu_entry_index is None:
            return None
        return options[menu_entry_index]

    def load_dataframe(self):
        filename = self.select_from_options(os.listdir(self.stats_path))
        if filename is None:
            return None
        filepath = os.path.join(self.stats_path, filename)

        try:
            df = pd.read_csv(filepath)

            # Scale latencies to milliseconds
            df['latency'] = df['latency'].apply(lambda x: 1000.0*x)
            df['offload_latency'] = df['offload_latency'].apply(lambda x: 1000.0*x)

            df = df.rename(columns={ 'latency': 'Latency (ms)', 'offload_latency': 'Offload latency (ms)' })
            return df
        except FileNotFoundError:
            print(f"File '{filename}' was not found in directory '{STATS_PATH}'. Make sure that it exists and is readable.")

    def parse_boxplot_frame(self):
        df = self.load_dataframe()

        if df is None:
            return None

        no_datasources = input("Number of data sources in experiment: ")
        used_scheduling = self.select_from_options(["OTN", "None"])
        df = df.assign(Datasources=int(no_datasources), Scheduling=used_scheduling)

        # Statistics only for offloaded tasks
        df = df.loc[df['request_op']=='offload_task']

        return df

def print_statistics(df):

    # Calculate statistics for offloaded tasks
    df = df.loc[df['request_op']=='offload_task']

    # Print statistics
    print(df[['latency (ms)', 'offload_latency (ms)']].describe())

def plot_boxplot(file_loader):
    plt.figure(figsize=(8,4))

    frames = []
    while True:
        df = file_loader.parse_boxplot_frame()
        if df is None:
            break
        frames.append(df)

    ax = sns.boxplot(x="Datasources", y="Offload latency (ms)", hue="Scheduling", data=pd.concat(frames), palette="dark:grey")

    filename = "/var/lib/sparse/stats/scheduling_boxplot.png"
    plt.savefig(filename, dpi=400)
    print(f"Saved boxplot to '{filename}'")

if __name__ == "__main__":
    plot_boxplot(StatisticsFileLoader())
