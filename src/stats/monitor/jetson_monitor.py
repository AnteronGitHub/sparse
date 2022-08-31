from jtop import jtop
from jtop.core.exceptions import JtopException

from . import Monitor

class JetsonMonitor(Monitor):
    def get_metrics(self):
        return ['GPU', 'RAM', 'SWAP', 'power_cur']

    def get_stats(self):
        try:
            with jtop() as jetson:
                if jetson.ok():
                    stats = jetson.stats
                    return [jetson.stats['GPU'], jetson.stats['RAM'], jetson.stats['SWAP'], jetson.stats['power cur']]
        except JtopException:
            return [0, 0, 0, 0]