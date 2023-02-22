def run_monitor():
    from sparse.stats.monitor_server import MonitorServer
    MonitorServer().start()

if __name__ == '__main__':
    run_monitor()
