import modal
import time

app = modal.App("example-fanout")

time_per_unit = 10

@app.function()
def multiplier(my_input:int):
    time.sleep(time_per_unit)
    return my_input * 2



@app.local_entrypoint()
def main():
    job_count = 1000
    to = time.time()
    print(list(multiplier.map(range(job_count))))
    wallclock = time.time() - to
    compute_time = job_count * time_per_unit
    print(f"Wallclock: {wallclock}s, Compute: {compute_time}s")
    