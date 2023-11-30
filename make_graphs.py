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

    def load_dataframe(self, dataframe_type = None):
        if dataframe_type is None:
            dataframe_type = plotter.file_loader.select_from_options(["ClientRequestStatisticsRecord", "ServerRequestStatisticsRecord"], "Data frame type:")

        data_files = [path for path in os.listdir(self.stats_path) if path.startswith(dataframe_type) and path.endswith(".csv")]
        data_files.sort()
        filename = self.select_from_options(data_files, "Dataframe file name:")

        if filename is None:
            return None, None
        filepath = os.path.join(self.stats_path, filename)

        try:
            return dataframe_type, pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"File '{filename}' was not found in directory '{STATS_PATH}'. Make sure that it exists and is readable.")

    def parse_boxplot_frame(self):
        dataframe_type, df = self.load_dataframe()

        if df is None:
            return None

        no_datasources = input("Number of data sources in experiment: ")
        used_scheduling = self.select_from_options(["OTN", "None"], "Scheduling method:")
        df = df.assign(Connections=int(no_datasources), Scheduling=used_scheduling)

        # Statistics only for offloaded tasks
        df = df.loc[df['request_op']=='offload_task']

        return df

    def parse_scales(self):
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

        return title, start_at, end_at, y_min, y_max

class StatisticsGraphPlotter:
    def __init__(self, write_path = '/var/lib/sparse/stats'):
        self.file_loader = StatisticsFileLoader()
        self.write_path = write_path

    def count_offload_task_client_statistics(self, df, start_at = 0.0, end_at = -1.0):
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

    def count_offload_task_server_statistics(self, df, start_at = 30.0, end_at = -1.0):
        df = df.loc[df['request_op']=='offload_task']

        df['e2e_latency'] = df['response_sent_at'] - df['request_received_at']
        df['rx_latency'] = df['task_queued_at'] - df['request_received_at']
        df['queueing_time'] = df['task_started_at'] - df['task_queued_at']
        df['task_latency'] = df['task_completed_at'] - df['task_started_at']
        df['tx_latency'] = df['response_sent_at'] - df['task_completed_at']

        # Scale latencies to milliseconds
        for column in ['e2e_latency', 'rx_latency', 'queueing_time', 'task_latency', 'tx_latency']:
            df[column] = df[column].apply(lambda x: 1000.0*x)

        # Rename columns
        df = df.rename(columns={ 'e2e_latency': 'Service time (ms)',
                                 'rx_latency': 'RX latency (ms)',
                                 'queueing_time': 'Queueing time (ms)',
                                 'task_latency': 'Task latency (ms)',
                                 'tx_latency': 'TX latency (ms)' })

        # Translate time axis
        df = df[df["request_received_at"] >= start_at]
        if end_at > 0.0:
            df = df[df["request_received_at"] <= end_at]
        df["request_received_at"] = df["request_received_at"].apply(lambda x: x-start_at)

        return df.set_index("request_received_at")

    def print_statistics(self):
        dataframe_type, df = self.file_loader.load_dataframe()

        if dataframe_type == "ClientRequestStatisticsRecord":
            # Calculate statistics for offloaded tasks
            stats = self.count_offload_task_client_statistics(df)
            # Print statistics
            print(stats[['Latency (ms)', 'Offload latency (ms)']].describe())
        else:
            stats = self.count_offload_task_server_statistics(df)
            print(stats[['Service time (ms)', 'RX latency (ms)', 'Queueing time (ms)', 'Task latency (ms)', 'TX latency (ms)']].describe())

    def plot_latency_timeline(self):
        dataframe_type, df = self.file_loader.load_dataframe()
        title, start_at, end_at, y_min, y_max = self.file_loader.parse_scales()

        marker = 'o' if (input("Use marker in plot points (y/N): ")) == "y" else ''

        # Count analytics
        if dataframe_type == "ClientRequestStatisticsRecord":
            stats = self.count_offload_task_client_statistics(df, start_at, end_at)
        else:
            stats = self.count_offload_task_server_statistics(df, start_at, end_at)

        # Plot graph
        ylabel = "Offload latency (ms)" if dataframe_type == "ClientRequestStatisticsRecord" else "Service time (ms)"
        xlabel = "Request sent (s)" if dataframe_type == "ClientRequestStatisticsRecord" else "Request received (s)"
        fig, ax = plt.subplots(figsize=(12,6))
        for label, data in stats.groupby("connection_id"):
            data.plot(y=ylabel, ax=ax, label=label, marker=marker)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.grid()

        ax.get_legend().remove()
        if end_at > 0:
            ax.set_xlim([0, end_at - start_at])

        if y_min > 0 and y_max > 0:
            ax.set_ylim([y_min, y_max])

        filepath = os.path.join(self.write_path, f"{dataframe_type}_latency_timeline.png")
        plt.savefig(filepath, dpi=600)
        print(f"Saved column plot to '{filepath}'")

    def plot_benchmark_barplot(self):
        title, start_at, end_at, y_min, y_max = self.file_loader.parse_scales()

        plt.figure(figsize=(8,4))

        barplot_data = pd.DataFrame([], columns=["Connections", "RX", 'Queueing', 'Task', "TX"])
        dataframe_type = "ServerRequestStatisticsRecord"
        while True:
            _, df = self.file_loader.load_dataframe(dataframe_type)
            if df is None:
                break

            no_connections = int(input("Number of connections: "))
            stats = self.count_offload_task_server_statistics(df)

            barplot_data.loc[len(barplot_data.index)] = [no_connections,
                                                         stats.loc[:, 'RX latency (ms)'].mean(),
                                                         stats.loc[:, 'Queueing time (ms)'].mean(),
                                                         stats.loc[:, 'Task latency (ms)'].mean(),
                                                         stats.loc[:, 'TX latency (ms)'].mean()]

        barplot_data = barplot_data.set_index("Connections")

        ax = barplot_data.plot.bar(rot=0, stacked=True)

        plt.title(title)
        plt.ylabel("Latency (ms)")

        if y_max > 0:
            ax.set_ylim([0, y_max])

        filepath = os.path.join(self.write_path, f"{dataframe_type}_latency_barplot.png")
        plt.savefig(filepath, dpi=400)
        print(f"Saved benchmark barplot to '{filepath}'")

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

            frames.append(self.count_offload_task_client_statistics(df, start_at, end_at))

        ax = sns.boxplot(x="Connections",
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
    operation = plotter.file_loader.select_from_options(["Print statistics", "Plot timeline", "Plot barplot", "Plot boxplot"], "Select operation:")
    if operation == "Print statistics":
        plotter.print_statistics()
    elif operation == "Plot timeline":
        plotter.plot_latency_timeline()
    elif operation == "Plot barplot":
        plotter.plot_benchmark_barplot()
    else:
        plotter.plot_offload_latency_boxplot()
