import modal

app = modal.App()
image = modal.Image.debian_slim().pip_install("torch")

@app.function(gpu="A100", image=image)
def run():
    import torch
    if torch.cuda.is_available():
        print("GPU is available!")
    else:
        print("GPU is not available")

if __name__ == "__main__":
    run.remote()
