import modal
import open_clip
import torch
import os
import time
from PIL import Image
import requests

image = modal.Image.debian_slim().pip_install_from_pyproject("pyproject.toml")
app = modal.App("feature-extract-demo")
volume = modal.Volume.from_name("clip-image-embeddings", create_if_missing=True)


@app.cls(image=image, volumes={"/data": volume})
class ClipBatchExtractor:
    def __init__(self):
        self.model = None
        self.preprocess = None

    @modal.enter()
    def load_model(self):
        model_path = "/data/clip_model.pt"
        if os.path.exists(model_path):
            # Load model from volume
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=model_path)
        else:
            # Download and save model
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            torch.save(model.state_dict(), model_path)
        self.model, self.preprocess = model, preprocess

    @modal.method()
    def embed(self,key: int):        
        t0 = time.time()
        image_url = f"https://picsum.photos/200?{key}"
        image = Image.open(requests.get(image_url, stream=True).raw)
        image = self.preprocess(image).unsqueeze(0)
        print(f"Image loaded in {time.time() - t0} seconds")

        t0 = time.time()
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        print(f"Image features extracted in {time.time() - t0} seconds")
        
        path = f"/data/features/{key}.pt"
        os.makedirs("/data/features", exist_ok=True)
        torch.save(image_features, path)

        
    

@app.local_entrypoint()
def main():
    ClipBatchExtractor().embed.spawn_map(range(100))


