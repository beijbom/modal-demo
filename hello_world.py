
import modal

# During the demo
# Add a volume to the app. Store some data

app = modal.App("example-hello-world")

@app.function()
def main():
    print("Hello from the cloooooud!")


if __name__ == "__main__":
    main.remote()
