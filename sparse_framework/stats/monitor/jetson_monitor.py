from jtop import jtop
from jtop.core.exceptions import JtopException

from . import Monitor

class JetsonMonitor(Monitor):
    def get_metrics(self):
        return ['GPU1', 'RAM', 'SWAP', 'power_cur']

    def get_stats(self):
        try:
            with jtop() as jetson:
                if jetson.ok():
                    stats = jetson.stats
                    return [jetson.stats['GPU1'], jetson.stats['RAM'], jetson.stats['SWAP'], jetson.stats['power cur']]
        except JtopException as e:
            return [0, 0, 0, 0]
