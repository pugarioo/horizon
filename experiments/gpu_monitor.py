import csv
import threading
import time

import pynvml


class GPUMonitor:
    def __init__(self, interval=1.0, output_file="gpu_metrics.csv"):
        self.interval = interval
        self.output_file = output_file
        self.is_running = False

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def _poll(self):
        # Open file stream once and keep open during execution
        with open(self.output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "GPU_Util(%)", "VRAM_Used(MB)"])

            while self.is_running:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

                # Write and flush immediately to disk
                writer.writerow([time.time(), util.gpu, mem.used / (1024**2)])
                file.flush()
                time.sleep(self.interval)

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._poll)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.thread.join()
        pynvml.nvmlShutdown()
