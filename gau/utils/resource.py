import torch
import time
import logging
import subprocess

class ResourceMonitor:
    def __init__(self, log_file="resource_log.txt"):
        self.start_time = time.time()
        self.max_memory = 0
        self.log_file = log_file
        with open(self.log_file, "w") as f:
            f.write("Resource Log\n")

    def log_cpu_memory_disk():
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        logging.info(f"CPU: {cpu}%, Memory: {mem}%, Disk: {disk}%")
        return cpu, mem, disk

    def log_gpu_memory(self, step=None):
        mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        self.max_memory = max(self.max_memory, mem)
        msg = f"Step {step}: Max GPU memory allocated: {mem:.2f} GB"
        logging.info(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

    def log_time(self, message="Elapsed time"):
        elapsed = time.time() - self.start_time
        msg = f"{message}: {elapsed:.2f} seconds"
        logging.info(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

    def log_gpu_utilization(self):
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"]
            )
            util = result.decode("utf-8").strip()
            msg = f"Current GPU utilization: {util}%"
            logging.info(msg)
            with open(self.log_file, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            logging.warning(f"Could not query GPU utilization: {e}")

    def summary(self):
        self.log_time("Total run time")
        msg = f"Peak GPU memory usage: {self.max_memory:.2f} GB"
        logging.info(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")
