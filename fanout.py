import modal
import time

app = modal.App("example-fanout")

time_per_job_seconds = 10

@app.function()
def my_job(job_id: int):
    t0 = time.time()
    print(f"Job {job_id} started")
    time.sleep(time_per_job_seconds)
    t1 = time.time()
    print(f"Job {job_id} finished in {t1 - t0} seconds")
    return t1 - t0

@app.local_entrypoint()
def main():
    job_count = 100
    to = time.time()
    cpu_clock = sum(list(my_job.map(range(job_count))))
    wall_clock = time.time() - to
    print(f"Wallclock: {wall_clock}s, Compute: {cpu_clock}s, Speedup: {wall_clock / cpu_clock}")
    