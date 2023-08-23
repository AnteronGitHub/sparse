import asyncio
import time

class ClockedLoop:
    def __init__(self, update_frequency_ps = 8):
        self.update_frequency_ps = update_frequency_ps

    def loop_task(self):
        pass

    async def run_clocked_loop(self):
        while True:
            start_time = time.time()
            self.loop_task()
            time_elapsed = time.time() - start_time
            await asyncio.sleep(1.0/self.update_frequency_ps - time_elapsed)

