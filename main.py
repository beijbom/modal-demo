
import modal

app = modal.App("example-hello-world")

@app.function()
def main():
    print("Hello from the cloooooud!")


if __name__ == "__main__":
    main.remote()
