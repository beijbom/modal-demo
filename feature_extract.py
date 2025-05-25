import os
import time

import modal
import open_clip
import requests
import torch
from PIL import Image

image = modal.Image.debian_slim().pip_install_from_pyproject("pyproject.toml")
app = modal.App("openclip-demo")
volume = modal.Volume.from_name("clip-image-embeddings", create_if_missing=True)

@app.cls(image=image, volumes={"/data": volume}, retries=2)
class ClipBatchExtractor:
    @modal.enter()
    def load_model(self):
        os.makedirs("/data/features", exist_ok=True)
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
        path = f"/data/features/{key}.pt"
        if os.path.exists(path):
            print(f"Feature file {path} already exists")
            return       
        t0 = time.time()
        image_url = f"https://picsum.photos/200?{key}"
        image = Image.open(requests.get(image_url, stream=True).raw)
        image = self.preprocess(image).unsqueeze(0)
        print(f"Image loaded in {time.time() - t0} seconds")

        t0 = time.time()
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        print(f"Image features extracted in {time.time() - t0} seconds")
        torch.save(image_features, path)

    @modal.method()
    def inspect(self):
        path = f"/data/features/"
        feature_file_count = len(os.listdir(path))
        print(f"Found {feature_file_count} feature files")
        one_feature_file = os.listdir(path)[0]
        feature = torch.load(os.path.join(path, one_feature_file))
        print(f"Feature shape: {feature.shape}")
    

@app.local_entrypoint()
def main(job_name: str):
    if job_name == "submit":
        ClipBatchExtractor().embed.spawn_map(range(100_000))
    elif job_name == "inspect":
        ClipBatchExtractor().inspect.remote()


