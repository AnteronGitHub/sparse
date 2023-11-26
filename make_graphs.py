import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from simple_term_menu import TerminalMenu

pd.options.mode.chained_assignment = None

class StatisticsFileLoader:
    def __init__(self, stats_path = '/var/lib/sparse/stats'):
        self.stats_path = stats_path

    def select_from_options(self, options, title) -> str:
        terminal_menu = TerminalMenu(options, title=title)
        menu_entry_index = terminal_menu.show()
        if menu_entry_index is None:
            return None
        print(f"{title} {options[menu_entry_index]}")
        return options[menu_entry_index]

    def load_dataframe(self):
        filename = self.select_from_options([path for path in os.listdir(self.stats_path) if path.endswith(".csv")],
                                            "Dataframe file name:")
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

    def count_offload_task_statistics(self, df, start_at = 0.0, end_at = -1.0):
        """Calculates latencies from a Dataframe with time stamps, and returns a new DataFrame with the results.

        Result DataFrame uses 'request_sent_at' timestamp as the index.
        """
        df = df.loc[df['request_op']=='offload_task']

        df['e2e_latency'] = df['response_received_at'] - df['processing_started_at']
        df['offload_latency'] = df['response_received_at'] - df['request_sent_at']

        # Scale latencies to milliseconds
        df['e2e_latency'] = df['e2e_latency'].apply(lambda x: 1000.0*x)
        df['offload_latency'] = df['offload_latency'].apply(lambda x: 1000.0*x)

        df = df.rename(columns={ 'e2e_latency': 'Latency (ms)', 'offload_latency': 'Offload latency (ms)' })

        # Drop first rows to ignore 'slow start'
        # (See e.g: https://stackoverflow.com/questions/64420777/opencv-cuda-api-very-slow-at-the-first-call)
        df = df[df["request_sent_at"] >= start_at]
        if end_at > 0.0:
            df = df[df["request_sent_at"] <= end_at]
        df["request_sent_at"] = df["request_sent_at"].apply(lambda x: x-start_at)

        return df.set_index("request_sent_at")

    def print_statistics(self):
        df = self.file_loader.load_dataframe()

        # Calculate statistics for offloaded tasks
        stats = self.count_offload_task_statistics(df)

        # Print statistics
        print(stats[['Latency (ms)', 'Offload latency (ms)']].describe())

    def plot_offload_latency_timeline(self):
        # Parse arguments and load data frame
        df = self.file_loader.load_dataframe()
        title = input("Plot title: ")
        try:
            start_at = float(input("Start at timestamp: "))
        except ValueError:
            start_at = 0.0
        try:
            end_at = float(input("End at timestamp: "))
        except ValueError:
            end_at = -1.0
        try:
            y_min = float(input("Min y: "))
        except ValueError:
            y_min = -1
        try:
            y_max = float(input("Max y: "))
        except ValueError:
            y_max = -1

        marker = 'o' if (input("Use marker in plot points (y/N): ")) == "y" else ''

        # Count analytics
        stats = self.count_offload_task_statistics(df, start_at, end_at)

        # Plot graph
        fig, ax = plt.subplots(figsize=(12,6))
        for label, data in stats.groupby("node_id"):
            data.plot(y="Offload latency (ms)", ax=ax, label=label, marker=marker)

        plt.title(title)
        plt.ylabel("Offload latency (ms)")
        plt.xlabel("Request sent (s)")
        plt.grid()

        ax.get_legend().remove()
        if end_at > 0:
            ax.set_xlim([0, end_at - start_at])

        if y_min > 0 and y_max > 0:
            ax.set_ylim([y_min, y_max])

        filepath = os.path.join(self.write_path, "offload_latency_timeline.png")
        plt.savefig(filepath, dpi=600)
        print(f"Saved column plot to '{filepath}'")

    def plot_offload_latency_boxplot(self):
        plt.figure(figsize=(8,4))

        frames = []
        while True:
            df = self.file_loader.parse_boxplot_frame()
            if df is None:
                break
            try:
                start_at = float(input("Start at timestamp: "))
            except ValueError:
                start_at = 0.0
            try:
                end_at = float(input("End at timestamp: "))
            except ValueError:
                end_at = -1.0

            frames.append(self.count_offload_task_statistics(df, start_at, end_at))

        ax = sns.boxplot(x="Datasources",
                         y="Offload latency (ms)",
                         hue="Scheduling",
                         data=pd.concat(frames),
                         palette="dark:grey",
                         showfliers=False)

        filepath = os.path.join(self.write_path, "offload_latency_boxplot.png")
        plt.savefig(filepath, dpi=400)
        print(f"Saved offload latency boxplot to '{filepath}'")

if __name__ == "__main__":
    plotter = StatisticsGraphPlotter()
    plot_type = plotter.file_loader.select_from_options(["Timeline", "Boxplot"], "Select graph to plot:")
    if plot_type == "Timeline":
        plotter.plot_offload_latency_timeline()
    else:
        plotter.plot_offload_latency_boxplot()
