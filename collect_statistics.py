from datetime import datetime
import importlib
import os
import psutil
import time

def read_jetson_stats(stats):
    curr_time = int(stats['time'].timestamp() * 1000000000)
    # todo: Replace RAM usage with relative value
    gpu_util = stats['GPU']
    ram_usage = stats['RAM']
    swap_usage = stats['SWAP']
    power_cur = stats['power cur']

    return curr_time, gpu_util, ram_usage, swap_usage, power_cur

def read_network_stats(if_name):
    nic_stats = psutil.net_io_counters(pernic=True)
    [bytes_sent, bytes_recv, packets_sent, packets_recv, errin, errout, dropin, dropout] = nic_stats[if_name]
    return bytes_sent, bytes_recv

def collect_statistics_jtop(filehandle, if_name):
    header_row = 'timestamp,gpu_util,ram_usage,swap_usage_mbytes,power_cur,bytes_sent,bytes_recv'
    print(header_row)
    f.write(header_row + '\n')

    initial_time = stats = initial_bytes_sent = initial_bytes_recv = None
    with jtop() as jetson:
        # jetson.ok() will provide the proper update frequency
        while jetson.ok():
            try:
                # Read new measurements
                bytes_sent, bytes_recv = read_network_stats(if_name)
                curr_time, gpu_util, ram_usage, swap_usage, power_cur = read_jetson_stats(jetson.stats)

                if initial_time == None:
                    initial_time = curr_time
                    initial_bytes_sent = bytes_sent
                    initial_bytes_recv = bytes_recv

                row = f"{curr_time-initial_time},{gpu_util},{ram_usage},{swap_usage},{power_cur},{bytes_sent-initial_bytes_sent},{bytes_recv-initial_bytes_recv}"

                print(row)
                f.write(row + '\n')

            except KeyboardInterrupt:
                print("Stopping due to keyboard interrupt")
                break

def collect_statistics_general(filehandle, if_name):
    header_row = 'timestamp,bytes_sent,bytes_recv'
    print(header_row)
    f.write(header_row + '\n')

    initial_time = initial_bytes_sent = initial_bytes_recv = None
    while True:
        try:
            # Read new measurements
            curr_time = time.time_ns()
            bytes_sent, bytes_recv = read_network_stats(if_name)
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

            # Wait for a while
            time.sleep(1)

        except KeyboardInterrupt:
            print("Stopping due to keyboard interrupt")
            break

if __name__ == "__main__":
    print("========================================================================")

    OUTPUT_DIR = './data/stats'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(OUTPUT_DIR, f'statistics-{experiment_time}.csv')
    print(f'Writing statistics to file {filepath!r}')

    IF_NAME = 'lo'
    print(f'Collecting network statistics from interface {IF_NAME!r}')

    jtop_available = importlib.util.find_spec("jtop") is not None
    if jtop_available:
        from jtop import jtop
    else:
        print("No jtop available, collecting only network statistics")

    print("========================================================================")

    with open(filepath, 'a') as f:
        if jtop_available:
            collect_statistics_jtop(f, IF_NAME)
        else:
            collect_statistics_general(f, IF_NAME)

