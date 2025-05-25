import modal

app = modal.App()
image = modal.Image.debian_slim().pip_install("torch", "numpy")


# Try using T4, A100, A100-80GB and A100-80GB:2 
@app.function(gpu="T4", image=image)
def run():
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"Free memory: {free / 1024 / 1024} MB")
        print(f"Total memory: {total / 1024 / 1024} MB")
        print(f"GPU count: {torch.cuda.device_count()}")
    else:
        print("GPU poor :(")

if __name__ == "__main__":
    run.remote()
